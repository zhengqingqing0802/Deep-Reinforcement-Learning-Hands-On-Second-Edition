#!/usr/bin/env python3
import gym
import time
import numpy as np

import torch
import torch.nn as nn

from multiprocessing import Pool, cpu_count

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

class Individual:

    def __init__(self, env_name, nhid):

        self.env = gym.make(env_name)
        self.net = Net(self.env.observation_space.shape[0], self.env.action_space.shape[0], nhid)
        self.fitness = None

    def eval(self):

        obs = self.env.reset()
        self.fitness = 0
        steps = 0
        while True:
            obs_v = torch.FloatTensor([obs])
            action_v = self.net(obs_v.type(torch.FloatTensor))
            obs, r, done, _ = self.env.step(action_v.data.numpy()[0])
            self.fitness += r
            steps += 1
            if done:
                break


def mutate_net(net, noise_std, seed):
    if seed is not None:
        np.random.seed(seed)
    for p in net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        p.data += noise_std * noise_t

def build_net(env, nhid, noise_std, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return Net(env.observation_space.shape[0], env.action_space.shape[0], nhid)

if __name__ == "__main__":

    MAX_SEED = 2**32 - 1

    #parser = make_ga_parser("Pendulum-v0", 64, 2000, 0.01)
    parser = make_ga_parser("Pendulum-v0", 64, 10, 0.01)

    args = parser.parse_args()

    workers_count = cpu_count()

    writer = SummaryWriter(comment=args.env)

    # XXX needed?
    if args.seed is not None:
        np.random.seed(0)

    # Create initial population
    population = [Individual(args.env, args.hid) for _ in range(args.population_size)]  

    gen_idx = 0

    elite = None

    while True:

        t_start = time.time()

        batch_steps = 0

        for p in population:
            p.eval()
            print(p.fitness)

        #population.sort(key=lambda p: p.fitness, reverse=True)

        break

    '''
        population = []
        while len(population) < seeds_per_worker * workers_count:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)
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
    '''
