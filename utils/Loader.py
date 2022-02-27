import json
import os
from datetime import datetime

import yaml

from algos.DQN import DQN_agent
from algos.PPO import PPO_agent
from algos.SPR import SPR_agent
from utils.logger import logger

# TODO: move constants to certain file
DEFAULT_CONFIG_PATH = "configs/default.yaml"


def read_config(path: str):
    with open(path, "r") as f:
        if path.endswith("json"):
            return json.load(f)
        elif path.endswith("yaml"):
            return yaml.safe_load(f)


def write_config(path: str, config: dict):
    with open(path, "w") as f:
        if path.endswith("json"):
            f.write(json.dumps(config))
        elif path.endswith("yaml"):
            yaml.dump(config, f)


def load_agent(
    cfg_path, run_name, ckpt_folder="checkpoints", config=dict(), override=False
):
    if run_name is None:
        tconf = read_config(cfg_path)
        run_name = "{}-{}-{}".format(
            tconf["env_name"],
            "-".join(tconf["algo"]),
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        )
    if not os.path.exists(os.path.join(ckpt_folder, run_name)) or override:
        logger.warning("Creating new run")
        fconf = read_config(cfg_path)
        dconf = read_config(DEFAULT_CONFIG_PATH)
        for key, val in dconf.items():
            if key not in fconf:
                fconf[key] = val
        for key, val in fconf.items():
            if key not in config:
                config[key] = val
        config["time"] = str(datetime.now())
        config["ckpt_folder"] = ckpt_folder
        config["run_name"] = run_name
        os.makedirs(os.path.join(ckpt_folder, run_name), exist_ok=override)
        write_config(os.path.join(ckpt_folder, run_name, "config.yaml"), config)
    else:
        logger.info("Loading existing run")
        try:
            fconf = read_config(os.path.join(ckpt_folder, run_name, "config.json"))
        except FileNotFoundError:
            fconf = read_config(os.path.join(ckpt_folder, run_name, "config.yaml"))
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
