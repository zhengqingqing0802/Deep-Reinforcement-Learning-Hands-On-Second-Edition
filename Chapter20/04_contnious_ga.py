#!/usr/bin/env python3
import gym
import collections
import copy
import time
import numpy as np
import os

import torch
import torch.multiprocessing as mp

from lib import parse_with_max_gen, Net

from tensorboardX import SummaryWriter

# Common classes and data structures -------------------------------

OutputItem = collections.namedtuple('OutputItem', field_names=['seeds', 'reward', 'steps'])

# Worker code -------------------------------------------------------

def evaluate(args):
    net, env, env_seed = args
    if env_seed is not None:
        env.seed(env_seed)
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        action_v = net(obs_v.type(torch.FloatTensor))
        obs, r, done, _ = env.step(action_v.data.numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def mutate_net(net, seed, noise_std, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        p.data += noise_std * noise_t
    return new_net


def build_net(env, seeds, nhid, noise_std):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape[0],
              env.action_space.shape[0], nhid)
    for seed in seeds[1:]:
        net = mutate_net(net, seed, noise_std, copy_net=False)
    return net

def save_net(net):
    fname = 'best.net'
    print('Saving ' + fname)

def worker_func(worker_id, cmdargs, main_to_worker_queue, worker_to_main_queue, noise_std, save_path):

    env = gym.make(cmdargs.env)
    cache = {}

    # Loop over generations, getting parent indicies from main process and mutating to get new population
    for _ in range(cmdargs.max_gen):
        parents = main_to_worker_queue.get()
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1], noise_std)
                else:
                    net = build_net(env, net_seeds, cmdargs.hid, noise_std)
            else:
                net = build_net(env, net_seeds, cmdargs.hid, noise_std)
            new_cache[net_seeds] = net
            reward, steps = evaluate((net, env, cmdargs.seed))
            worker_to_main_queue.put(OutputItem(seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache

    # Write best net to file if indicated
    if save_path is not None:
        nets = list(cache.values())
        rewards = [pair[0] for pair in [evaluate((net, env, cmdargs.seed)) for net in nets]]
        fname = '%s/%04d-best%+f.net' % (save_path, worker_id, max(rewards))
        best_net = nets[np.argmax(rewards)]
        torch.save(best_net, fname)

# Main code ----------------------------------------------------------

def report(writer, population, parents_count, gen_idx, batch_steps, t_start):

    rewards = [p[1] for p in population[:parents_count]]
    reward_mean = np.mean(rewards)
    reward_max = np.max(rewards)
    reward_std = np.std(rewards)
    writer.add_scalar("reward_mean", reward_mean, gen_idx)
    writer.add_scalar("reward_std", reward_std, gen_idx)
    writer.add_scalar("reward_max", reward_max, gen_idx)
    writer.add_scalar("batch_steps", batch_steps, gen_idx)
    writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
    speed = batch_steps / (time.time() - t_start)
    writer.add_scalar("speed", speed, gen_idx)
    print("%04d: reward_mean=%+6.2f\treward_max=%+6.2f\treward_std=%6.2f\tspeed=%d f/s" % (
        gen_idx, reward_mean, reward_max, reward_std, int(speed)))

def get_new_population(worker_to_main_queue, seeds_per_worker, workers_count):

    batch_steps = 0
    population = []
    while len(population) < seeds_per_worker * workers_count:
        out_item = worker_to_main_queue.get()
        population.append((out_item.seeds, out_item.reward))
        batch_steps += out_item.steps
    return population, batch_steps


def setup_workers(cmdargs, workers_count, seeds_per_worker, max_seed, save_path):

    main_to_worker_queues = []
    worker_to_main_queue = mp.Queue(workers_count)
    workers = []
    for k in range(workers_count):
        main_to_worker_queue = mp.Queue()
        main_to_worker_queues.append(main_to_worker_queue)
        w = mp.Process(target=worker_func, 
                args=(k, cmdargs, main_to_worker_queue, worker_to_main_queue, cmdargs.noise_std, save_path))
        workers.append(w)
        w.start()
        seeds = [(np.random.randint(max_seed),) for _ in range(seeds_per_worker)]
        main_to_worker_queue.put(seeds)
    return main_to_worker_queues, worker_to_main_queue, workers

def update_workers(population, main_to_worker_queues, seeds_per_worker, max_seed, parents_count):

    for main_to_worker_queue in main_to_worker_queues:
        seeds = []
        for _ in range(seeds_per_worker):
            parent = np.random.randint(parents_count)
            next_seed = np.random.randint(max_seed)
            s = list(population[parent][0]) + [next_seed]
            seeds.append(tuple(s))
        main_to_worker_queue.put(seeds)

def make_save_path(name):

    save_path  = None
    if name is not None:
        save_path = os.path.join("saves", "%s" % name)
        os.makedirs(save_path, exist_ok=True)
    return save_path

def main():

    # XXX could this be automated?
    MAX_SEED = 2**32 - 1

    # Get command-line args
    args = parse_with_max_gen("Pendulum-v0", 64, 2000, 0.01)

    # Make save directory if indicated
    save_path = make_save_path(args.name)

    # Use all available CPUs, distributing the population equally among them
    workers_count = mp.cpu_count()
    seeds_per_worker = args.pop_size // workers_count

    # Set up a TensorFlow summary writer using the environment name
    writer = SummaryWriter(comment=args.env)

    # Seed random-number generator if indicated
    if args.seed is not None:
        np.random.seed(0)

    # Set up communication with workers
    main_to_worker_queues, worker_to_main_queue, workers = setup_workers(args, workers_count, seeds_per_worker, MAX_SEED, save_path)

    # This will store the fittest individual in the population
    best = None

    # Loop for specified number of generations (default = inf)
    for gen_idx in range(args.max_gen):

        # Start timer for performance tracking
        t_start = time.time()

        # Get results (seeds and rewards) from workers
        population, batch_steps = get_new_population(worker_to_main_queue, seeds_per_worker, workers_count)

        # Keep the current best in the population
        if best is not None:
            population.append(best)

        # Sort population by reward (fitness)
        population.sort(key=lambda p: p[1], reverse=True)
        
        # Report and store current state
        report(writer, population, args.parents_count, gen_idx, batch_steps, t_start)

        # Get new best
        best = population[0]

        # Send new random seeds to wokers
        update_workers(population, main_to_worker_queues, seeds_per_worker, MAX_SEED, args.parents_count)

    # Done; shut down workers
    for w in workers:
        w.join()

if __name__ == "__main__":
    main()

