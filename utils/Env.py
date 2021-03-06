import importlib

import gym
import gym_snake


def get_env(env_name, use_raw=False):
    if env_name in ("tetris", "tetris-simple"):
        tetris = importlib.import_module("pytris-effect.src.gameui")
        return tetris.GameUI(
            graphic_mode=False,
            game_mode="JOURNEY",
            its_per_sec=5,
            sec_per_tick=0.5,
            debug=(env_name == "tetris-simple"),
        )
    else:  # for now, all other envs are openAI gym envs
        return gym.make(env_name).env if use_raw else gym.make(env_name)
