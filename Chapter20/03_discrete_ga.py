#!/usr/bin/env python3
import gym
import copy
import numpy as np

import torch
import torch.nn as nn

from lib import make_ga_parser

from tensorboardX import SummaryWriter

from multiprocessing import Pool, cpu_count

class Net(nn.Module):
    def __init__(self, obs_size, action_size, nhid):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, nhid),
            nn.ReLU(),
            nn.Linear(nhid, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(args):
    net, env = args
    obs = env.reset()
    reward = 0.0
    while True:
        obs_v = torch.FloatTensor([obs])
        act_prob = net(obs_v)
        acts = act_prob.max(dim=1)[1]
        obs, r, done, _ = env.step(acts.data.numpy()[0])
        reward += r
        if done:
            break
    return reward


def mutate_parent(net, noise_std):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise)
        p.data += noise_std * noise_t
    return new_net

def get_fitnesses(env, nets, seed):

    if seed is not None:
        env.seed(seed)

    with Pool(processes=cpu_count()) as pool:
        return list(pool.map(evaluate, zip(nets, [env]*len(nets))))

if __name__ == "__main__":

    parser = make_ga_parser("CartPole-v0", 32, 50, 0.01)

    args = parser.parse_args()
 
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    writer = SummaryWriter(comment=("-%s" % args.env))

    env = gym.make(args.env)

    gen_idx = 0
    nets = [
        Net(env.observation_space.shape[0], env.action_space.n, args.hid)
        for _ in range(args.population_size)
    ]

    fits =  get_fitnesses(env, nets, args.seed)

    population = list(zip(nets, fits))

    while True:
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:args.parents_count]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)

        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, "
              "reward_std=%.2f" % (
            gen_idx, reward_mean, reward_max, reward_std))
        if reward_mean > 199:
            print("Solved in %d steps" % gen_idx)
            break

        # generate next population
        prev_population = population
        nets = []
        for _ in range(args.population_size-1):
            parent_idx = np.random.randint(0, args.parents_count)
            parent = prev_population[parent_idx][0]
            net = mutate_parent(parent, args.noise_std)
            nets.append(net)
        gen_idx += 1

        fits =  get_fitnesses(env, nets, args.seed)

        population = [population[0]] + list(zip(nets, fits)) 

    writer.close()
