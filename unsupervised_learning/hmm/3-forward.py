#!/usr/bin/env python3

"""
This module initializes forward algorithm
for a hidden markov model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    initializes forward algorithm for a hidden
    markov model

    Observation - numpy.ndarray (T,) that contains
    index of the observation
        - T - number of observations
    Emission - numpy.ndarray (N, M) containing the
    emission probability of a specific observation
        given a hidden state
        - Emission[i, j] is the probability of observing
        j given the hidden state i
        - N is the number of hidden states
        - M is the number of all possible observations
    Transition - 2D numpy.ndarray (N, N) containing the
    transition probabilities
        - Transition[i, j] is the probability of transitioning
        from the hidden state i to j
    Initial - numpy.ndarray (N, 1) containing the probability
    of starting in a particular hidden state
    Return:
    P, F, or None, None on failure
        - P is the likelihood of the observations given the model
        - F is a numpy.ndarray (N, T) containing the forward path
        probabilities
        - F[i, j] is the probability of being in hidden state i
        after observing the first j observations
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    N1, N2 = Transition.shape
    if N1 != N or N2 != N:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    N3, N4 = Initial.shape
    if N3 != N or N4 != 1:
        return None, None
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        F[:, i] = np.sum(
            F[:, i - 1] * Transition.T * Emission[np.newaxis, :,
                                                  Observation[i]].T, axis=1)
    P = np.sum(F[:, -1])
    return P, F
