#!/usr/bin/env python3

"""
This module determines if markov chain
is absorbing"""

import numpy as np


def absorbing(P):
    """
    determines steady state probabilities
    of a markov chain

    P - square 2D numpy.ndarray: (n, n) -transition matrix
        - P[i, j] - probability of transitioning from
    state i to state j
        - n no. of states in the markov chain

    Returns: True if it is absorbing,
        or False on failure
    """
    # absorbing states are states that have a probability of 1
    # of transitioning to themselves
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    n, n = P.shape
    if n != P.shape[0]:
        return False
    if np.sum(P, axis=1).all() != 1:
        return False
    if np.any(np.diag(P) == 1):
        return True
    return False
