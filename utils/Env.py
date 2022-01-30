import gym
import gym_snake


def get_env(env_name, use_raw=False):
    if env_name == "tetris":
        raise NotImplementedError()
    else:  # for now, all other envs are openAI gym envs
        return gym.make(env_name).env if use_raw else gym.make(env_name)
