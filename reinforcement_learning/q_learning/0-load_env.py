#!/usr/bin/env python3
""" Load Frozen Lake Environment """

import gym
from gym.envs.toy_text import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from OpenAI Gym
    """

    if desc is not None:
        env = FrozenLakeEnv(desc=desc, is_slippery=is_slippery)
    elif map_name is not None:
        env = gym.make(f'FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    else:
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery)

    return env
