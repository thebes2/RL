import argparse
from mpi4py import MPI
import tensorflow as tf
import sys
import time
import json
import os
import importlib
import random
from PIL import Image

sys.path.insert(0, '..')
from algos.PPO import PPO_agent
from utils.mpi import broadcast_model, average_gradients

comm = MPI.COMM_WORLD
size = comm.Get_size()
rnk = comm.Get_rank()


tetris = importlib.import_module('pytris-effect.src.gameui')

run_name = 'tetris'
action = 'train'

cfg_fp = os.path.join('configs', run_name + '.json')
with open(cfg_fp, 'r') as f:
    config = json.load(f)
ckpt_folder = os.path.join('checkpoints')

env_name = config['env']
if run_name == 'tetris':
    env = tetris.GameUI(graphic_mode=False, its_per_sec=2, sec_per_tick=0.5)
else:
    env = gym.make(env_name).env if 'use_raw_env' in config else gym.make(env_name)

print(env.reset().shape)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(20,10,3)),
    tf.keras.layers.Conv2D(32, (2, 2), activation='elu', padding='same'),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='valid'), # new addition
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(7, activation='softmax') # NO-OP is an action
])
value = tf.keras.Sequential([
    tf.keras.Input(shape=(20,10,3)),
    tf.keras.layers.Conv2D(32, (2, 2), activation='elu', padding='same'),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='valid'), # new addition
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(1, activation=None)
])


agent = PPO_agent(
    model,
    value,
    env=env,
    learning_rate=config['learning_rate'],
    minibatch_size=config['minibatch_size'],
    epsilon=0.15,
    env_name=config['env_name'],
    run_name='tetris-test-11',
    ckpt_folder=ckpt_folder,
)

t_max = config['t_max']

if rnk == 0:
    agent.load_from_checkpoint()

print('starting training...')
agent.mpi_train(rnk, t_max=t_max, buf_size=5000)
