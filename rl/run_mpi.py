import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #### REMOVE THIS LINE WHEN CUDA CONFIG IS FIXED
import argparse
from mpi4py import MPI
import tensorflow as tf
import sys
import time
import json
import importlib
import random
import gym
from PIL import Image
from datetime import datetime

from .models import get_policy_architecture, get_value_architecture
sys.path.insert(0, '..')
from algos.PPO import PPO_agent
from utils.mpi import broadcast_model, average_gradients

envs = os.listdir('configs')
envs = list(map(lambda x: x[:-5], envs))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task', type=str, default='cartpole',
    choices=envs
)
parser.add_argument(
    '--action', type=str, default='train',
    choices=['train'] # this script only handles training for now
    # there is no point in running validation with multithreading
)
parser.add_argument(
    '--algo', type=str, default='PPO',
    choices=['VPG', 'PPO', 'DQN']
)
parser.add_argument('--run_name', type=str, default=None)

mpi = True 

if mpi:
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rnk = comm.Get_rank()


# tetris = importlib.import_module('pytris-effect.src.gameui')

args = parser.parse_args()
task = args.task
algo = args.algo
action = args.action
run_name = args.run_name

if run_name is None:
    now = datetime.now()
    run_name = "{}-{}-{}".format(task, algo, now.strftime('%H-%M-%S'))

cfg_fp = os.path.join('configs', task + '.json')
with open(cfg_fp, 'r') as f:
    config = json.load(f)
ckpt_folder = os.path.join('checkpoints')

env_name = config['env']
if task == 'tetris':
    env = tetris.GameUI(
        graphic_mode=False, 
        its_per_sec=8, 
        sec_per_tick=0.25,
        colour_mode='mono'
    )
else:
    env = gym.make(env_name).env if 'use_raw_env' in config else gym.make(env_name)

model, value = get_policy_architecture(env_name), get_value_architecture(env_name)

agent = PPO_agent(
    model,
    value,
    env=env,
    learning_rate=config['learning_rate'],
    minibatch_size=config['minibatch_size'],
    epsilon=0.05,
    gamma=0.9,
    env_name=config['env_name'],
    run_name=run_name,
    ckpt_folder=ckpt_folder,
)

t_max = config['t_max']

if mpi and rnk == 0:
    agent.load_from_checkpoint()
elif not mpi:
    agent.load_from_checkpoint()

print('starting training...')
if mpi:
    agent.mpi_train(rnk, t_max=t_max, buf_size=config.get('buf_size', 20000))
else:
    agent.train(t_max=t_max, buf_size=5000)
