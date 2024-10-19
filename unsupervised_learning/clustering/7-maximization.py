#!/usr/bin/env python3

"""
This module contains a function that calculates
maximization step in the EM algorithm for a GMM
"""

import numpy as np


def maximization(X, g):
    """
    initializes variables for a Gaussian Mixture Model

    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    g: numpy.ndarray (k, n) containing the posterior
        probabilities for each data point in each cluster

    return:
        - pi: numpy.ndarray (k,) containing the updated priors for each cluster
        - m: numpy.ndarray (k, d) has updated centroid means for each cluster
        - S: numpy.ndarray (k, d, d) updated cov matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    if n != g.shape[1]:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None
    pi = np.sum(g, axis=1) / n
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]
    S = np.zeros((k, d, d))
    for i in range(k):
        y = X - m[i]
        S[i] = np.dot(g[i] * y.T, y) / np.sum(g[i])
    return pi, m, S
