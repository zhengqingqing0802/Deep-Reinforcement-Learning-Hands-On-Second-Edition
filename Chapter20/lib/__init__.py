import argparse

def make_parser(default_env, default_hid):

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default=default_env, help=("Environment id, default=%s" % default_env))
    parser.add_argument("--hid", default=default_hid, type=int, help=("Hidden units, default=%d" % default_hid))
    parser.add_argument("--seed", default=None, type=int, help="Seed for random number generators, default=None")
    return parser
 
