import argparse

def parse_args(default_env, default_hid):

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default=default_env, help=("Environment id, default=%s" % default_env))
    parser.add_argument("--hid", default=default_hid, type=int, help=("Hidden units, default=%d" % default_hid))
    return parser.parse_args()
 
