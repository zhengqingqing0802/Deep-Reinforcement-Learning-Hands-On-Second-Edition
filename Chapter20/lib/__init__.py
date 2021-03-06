import argparse
import numpy as np
import torch.nn as nn

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

def make_parser(default_env, default_hid):

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', default=default_env, help=('Environment id, default=%s' % default_env))
    parser.add_argument('--hid', default=default_hid, type=int, help=('Hidden units, default=%d' % default_hid))
    return parser

def make_ga_parser(default_env, default_hid, default_popsize, default_noise_std):

    parser = make_parser(default_env, default_hid)
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generators, default=None')
    parser.add_argument('--noise-std', type=float, default=default_noise_std)
    parser.add_argument('--pop-size', type=int, default=default_popsize)
    parser.add_argument('--parents-count', type=int, default=10)
    return parser
 
def parse_with_max_gen(default_env, default_hid, default_popsize, default_noise_std):

    parser = make_ga_parser(default_env, default_hid, default_popsize, default_noise_std)
    parser.add_argument('--max-gen', default=None, type=int, help='Maximum number of generations, default=inf')
    parser.add_argument('--name', required=False, help='Name of the run for saving')
    args = parser.parse_args()
    if args.max_gen is None:
        args.max_gen = np.iinfo(np.uint32).max
    return args
 
