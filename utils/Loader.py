import json
import os
from datetime import datetime

from algos.DQN import DQN_agent
from algos.PPO import PPO_agent
from algos.SPR import SPR_agent
from utils.logger import logger

# TODO: move constants to certain file
DEFAULT_CONFIG_PATH = "configs/default.json"


def load_agent(cfg_path, run_name, ckpt_folder="checkpoints", config=dict()):
    if run_name is None:
        with open(cfg_path, "r") as f:
            tconf = json.load(f)
        run_name = "{}-{}-{}".format(
            tconf["env_name"],
            "-".join(tconf["algo"]),
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        )
    if not os.path.exists(os.path.join(ckpt_folder, run_name)):
        logger.warning("Creating new run")
        with open(cfg_path, "r") as f:
            fconf = json.load(f)
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            dconf = json.load(f)
        for key, val in dconf.items():
            if key not in fconf:
                fconf[key] = val
        for key, val in fconf.items():
            if key not in config:
                config[key] = val
        config["time"] = str(datetime.now())
        config["ckpt_folder"] = ckpt_folder
        config["run_name"] = run_name
        os.makedirs(os.path.join(ckpt_folder, run_name))
        with open(os.path.join(ckpt_folder, run_name, "config.json"), "w") as f:
            f.write(json.dumps(config))
    else:
        logger.info("Loading existing run")
        with open(os.path.join(ckpt_folder, run_name, "config.json"), "r") as f:
            fconf = json.load(f)
        for key, val in fconf.items():
            if key not in config:
                config[key] = val

    logger.set_file_output(os.path.join(ckpt_folder, run_name, "logs.txt"))
    return load_from_config(config)


def load_from_config(config):
    logger.info("Loading agent: {}".format(config["run_name"]))
    if "SPR" in "\n".join(config["algo"]):
        agent = SPR_agent(config)
    # might need better way of doing this
    elif "DQN" in "\n".join(config["algo"]):
        agent = DQN_agent(config)
    elif "PPO" in config["algo"]:
        raise NotImplementedError("PPO_agent cannot load from cfg yet")
        agent = PPO_agent(config)
    return agent
