import argparse
import os
import sys

from utils.utils import *

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "-1"  #### REMOVE THIS LINE WHEN CUDA CONFIG IS FIXED


envs = os.listdir("configs")
envs = list(map(lambda x: x[:-5], envs))

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="cartpole", choices=envs)

parser.add_argument("--run_name", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    agent = load(args.run_name, args.env)
    agent.train()
