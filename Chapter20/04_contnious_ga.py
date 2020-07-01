#!/usr/bin/env python3
import gym
import collections
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from lib import make_ga_parser

from tensorboardX import SummaryWriter

class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size):
        super(Net, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mu(x)


def evaluate(args):
    net, env, seed = args
    if seed is not None:
        env.seed(seed)
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


OutputItem = collections.namedtuple(
    'OutputItem', field_names=['seeds', 'reward', 'steps'])


def worker_func(env_name, input_queue, output_queue, nhid, env_seed, noise_std):
    env = gym.make(env_name)
    cache = {}

    while True:
        parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = mutate_net(net, net_seeds[-1], noise_std)
                else:
                    net = build_net(env, net_seeds, nhid, noise_std)
            else:
                net = build_net(env, net_seeds, nhid, noise_std)
            new_cache[net_seeds] = net
            reward, steps = evaluate((net,env,env_seed))
            output_queue.put(OutputItem(
                seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache


if __name__ == "__main__":

    MAX_SEED = 2**32 - 1

    mp.set_start_method('spawn')

    parser = make_ga_parser("Pendulum-v0", 64, 2000)
    parser.add_argument("--workers-count", type=int, default=6)

    args = parser.parse_args()

    seeds_per_worker = args.population_size // args.workers_count

    writer = SummaryWriter(comment=args.env)

    if args.seed is not None:
        np.random.seed(0)

    input_queues = []
    output_queue = mp.Queue(maxsize=args.workers_count)
    workers = []
    for _ in range(args.workers_count):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        w = mp.Process(target=worker_func, args=(args.env, input_queue, output_queue, args.hid, args.seed, args.noise_std))
        w.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(seeds_per_worker)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None
    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < seeds_per_worker * args.workers_count:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:args.parents_count]]
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
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        for worker_queue in input_queues:
            seeds = []
            for _ in range(seeds_per_worker):
                parent = np.random.randint(args.parents_count)
                next_seed = np.random.randint(MAX_SEED)
                s = list(population[parent][0]) + [next_seed]
                seeds.append(tuple(s))
            worker_queue.put(seeds)
        gen_idx += 1

    pass
