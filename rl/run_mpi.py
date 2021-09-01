import argparse
from mpi4py import MPI
import tensorflow as tf
import sys
import time
import json
import os
import importlib
import random
import gym
from PIL import Image

sys.path.insert(0, '..')
from algos.PPO import PPO_agent
from utils.mpi import broadcast_model, average_gradients

mpi = True 

if mpi:
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
    env = tetris.GameUI(
        graphic_mode=False, 
        its_per_sec=8, 
        sec_per_tick=0.25,
        colour_mode='mono'
    )
else:
    env = gym.make(env_name).env if 'use_raw_env' in config else gym.make(env_name)

print(env.reset().shape)

if env_name == "CartPole-v0":
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    value = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)
    ])
elif env_name == "MountainCar-v0":
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
elif env_name == "Acrobot-v1":
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    value = tf.keras.Sequential([
        tf.keras.Input(shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)
    ])
elif env_name == "gym_snake:snake-v0":
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(15, 15, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    value = tf.keras.Sequential([
        tf.keras.Input(shape=(15, 15, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        #tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)
    ])
elif env_name == "tetris":  # the final raid boss
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(20,10,3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='valid'), # new addition
        tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(256, activation='elu'),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dense(7, activation='softmax') # NO-OP is an action
    ])
    value = tf.keras.Sequential([
        tf.keras.Input(shape=(20,10,3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='elu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='valid'), # new addition
        tf.keras.layers.Conv2D(128, (3, 3), activation='elu'),
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
    #run_name='tetris-test-mono2',
    ckpt_folder=ckpt_folder,
)

t_max = config['t_max']

if mpi and rnk == 0:
    agent.load_from_checkpoint()
elif not mpi:
    agent.load_from_checkpoint()

print('starting training...')
if mpi:
    agent.mpi_train(rnk, t_max=t_max, buf_size=5000)
else:
    agent.train(t_max=t_max, buf_size=5000)
