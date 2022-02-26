import argparse
import os

from utils.utils import *

envs = os.listdir("configs")
envs = list(map(lambda x: x[:-5], envs))

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default=None, choices=envs)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--override", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    if (
        "CUDA_VISIBLE_DEVICES" in os.environ
        and os.environ["CUDA_VISIBLE_DEVICES"] != "-1"
    ):
        print("Using gpu")
        import tensorflow as tf

        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    agent = load(args.run_name, args.env, override=args.override)
    agent.train()
