#!/usr/bin/env python3
""" Epsilon Greedy """

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon Greedy
    """

    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])
