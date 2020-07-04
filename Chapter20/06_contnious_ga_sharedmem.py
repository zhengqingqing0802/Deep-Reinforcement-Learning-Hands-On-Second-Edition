#!/usr/bin/env python3
import gym
import time
import numpy as np

import torch
import torch.nn as nn

from multiprocessing import Pool, cpu_count

from lib import parse_with_max_gen

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
        self.fit = None
        self.steps = None

    @staticmethod
    def eval(p):

        obs = p.env.reset()
        fit = 0
        steps = 0
        while True:
            obs_v = torch.FloatTensor([obs])
            action_v = p.net(obs_v.type(torch.FloatTensor))
            obs, r, done, _ = p.env.step(action_v.data.numpy()[0])
            fit += r
            steps += 1
            if done:
                break

        return fit, steps

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

def report(writer, pop, gen_idx, parents_count, t_start):
    batch_steps = np.sum([p.steps for p in pop])
    rewards = [p.fit for p in pop[:parents_count]]
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


def eval_fits(pop):
    with Pool(processes=cpu_count()) as pool:
        for p,fs in zip(pop, pool.map(Individual.eval, pop)):
            p.fit   = fs[0]
            p.steps = fs[1]

def main():

    #parser = make_ga_parser("Pendulum-v0", 64, 2000, 0.01)
    args = parse_with_max_gen("Pendulum-v0", 64, 10, 0.01)

    workers_count = cpu_count()

    writer = SummaryWriter(comment=args.env)

    # XXX needed?
    if args.seed is not None:
        np.random.seed(0)

    # Create initial population
    pop = [Individual(args.env, args.hid) for _ in range(args.pop_size)]  

    # Loop forever, or to maximum number of generations specified in command line
    for gen_idx in range(np.inf if args.max_gen is None else args.max_gen):

        t_start = time.time()

        # Evaulate fitnesses in parallel
        eval_fits(pop)

        # Sort population by fitness
        pop.sort(key=lambda p: p.fit, reverse=True)

        # Report everything
        report(writer, pop, gen_idx, args.parents_count, t_start)

        elite = pop[0]

if __name__ == "__main__":
    main()


