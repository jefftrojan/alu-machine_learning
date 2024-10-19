#!/usr/bin/env python3

"""
This module contains a function that
perfoms K-means on a dataset
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    perfoms K-means on a dataset

    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    k: positive integer - the no. of clusters
    iterations: +ve(int) - max no. of iterations perfomed

    return:
        - C: numpy.ndarray (k, d) containing the centroid
        for each cluster
        - clss: numpy.ndarray (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or type(k) is not int:
        return (None, None)
    if len(X.shape) != 2 or k < 0:
        return (None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None)
    n, d = X.shape
    if k == 0:
        return (None, None)
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        new_C = np.copy(C)
        for c in range(k):
            if c not in clss:
                new_C[c] = np.random.uniform(low, high)
            else:
                new_C[c] = np.mean(X[clss == c], axis=0)
        if (new_C == C).all():
            return (C, clss)
        else:
            C = new_C
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return (C, clss)
