#!/usr/bin/env python3

"""
This module contains a function that
calculates total intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """
    calculates intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    C: numpy.ndarray (k, d) containing the centroid
        for each cluster

    return:
        - var: total intra-cluster variance
    """
    var = np.sum((X - C[:, np.newaxis])**2, axis=-1)
    mean = np.sqrt(var)
    mini = np.min(mean, axis=0)
    var = np.sum(mini ** 2)
    return np.sum(var)
