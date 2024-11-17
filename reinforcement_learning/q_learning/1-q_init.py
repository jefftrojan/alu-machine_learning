#!/usr/bin/env python3
""" Initialize Q-Table """

import numpy as np


def q_init(env):
    """
    Initialize the Q-table
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    return np.zeros((n_states, n_actions))
