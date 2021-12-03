import json
from datetime import datetime
import os

from algos.DQN import DQN_agent
from algos.PPO import PPO_agent

# override saved configuration in cfg_path with config
def load_agent(cfg_path, run_name, ckpt_folder='checkpoints', config=dict()):
    if run_name is None:
        with open(cfg_path, 'r') as f:
            tconf = json.load(f)
        run_name = "{}-{}-{}".format(
            tconf['env_name'],
            "-".join(tconf['algo']),
            datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        )
    if not os.path.exists(os.path.join(ckpt_folder, run_name)):
        print("{}Creating new run{}".format('\033[93m', '\033[0m'))
        with open(cfg_path, 'r') as f:
            fconf = json.load(f)
        for key, val in fconf.items():
            if key not in config:
                config[key] = val
        config['time'] = str(datetime.now())
        config['ckpt_folder'] = ckpt_folder
        config['run_name'] = run_name
        os.makedirs(os.path.join(ckpt_folder, run_name))
        with open(os.path.join(ckpt_folder, run_name, 'config.json'), 'w') as f:
            f.write(json.dumps(config))
    else:
        print("{}Loading existing run{}".format('\033[92m', '\033[0m'))
        with open(os.path.join(ckpt_folder, run_name, 'config.json'), 'r') as f:
            config = json.load(f)
    return load_from_config(config)


def load_from_config(config):
    print("{}Loading agent: {}{}".format('\033[92m', config['run_name'], '\033[0m'))
    if 'DQN' in "\n".join(config['algo']): # might need better way of doing this
        return DQN_agent(config)
    elif 'PPO' in config['algo']:
        raise NotImplementedError("PPO_agent cannot load from cfg yet")
        return PPO_agent(config)
