import argparse
import json
import os
import sys
import time

import gym
import gym_snake
import matplotlib

from algos.PPO import PPO_agent

from .models import get_policy_architecture, get_value_architecture

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "-1"  #### REMOVE THIS LINE WHEN CUDA CONFIG IS FIXED

matplotlib.use("WX")


sys.path.insert(0, "..")

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="cartpole")
parser.add_argument("--run_name", type=str)
parser.add_argument("--t_max", type=int, default=10000)

args = parser.parse_args()
task = args.task
run_name = args.run_name
t_max = args.t_max

cfg_fp = os.path.join("configs", task + ".json")
with open(cfg_fp, "r") as f:
    config = json.load(f)
ckpt_folder = os.path.join("checkpoints")

env_name = config["env"]
env = gym.make(env_name).env if "use_raw_env" in config else gym.make(env_name)

policy, value = get_policy_architecture(env_name), get_value_architecture(env_name)

agent = PPO_agent(
    policy,
    value,
    env=env,
    env_name=config["env_name"],
    run_name=run_name,
    ckpt_folder=ckpt_folder,
)

agent.load_from_checkpoint()

obs = agent.preprocess(env.reset())
reward = 0
for i in range(t_max):
    # print(agent.get_policy(obs))
    act = agent.get_action(obs, greedy=True)[0]
    obs, r, dn, info = env.step(agent.action_wrapper(act))
    env.render()
    print(act, file=sys.stderr)
    time.sleep(0.05)
    obs = agent.preprocess(obs)
    reward += r
    if dn:
        break

print("Total reward: {}".format(reward), file=sys.stderr)
if close:
    env.close()
