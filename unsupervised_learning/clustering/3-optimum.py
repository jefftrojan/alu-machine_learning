#!/usr/bin/env python3

"""
This module contains a function that
tests for the optimum number of clusters by variance
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    calculates intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    kmin: positive integer - the minimum no. of clusters
    kmax: positive integer - the maximum no. of clusters
    iterations: +ve(int) - max no. of iterations perfomed

    return:
        - results: list containing the results of the
        K-means for each cluster size
        - d_vars: list containing the difference in variance
        from the smallest cluster size for each cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        if k == kmin:
            var = variance(X, C)
        d_vars.append(var - variance(X, C))
    return results, d_vars
