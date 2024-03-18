#!/usr/bin/env python3
"""Function that calculates the mean and covariance of a data set"""

import numpy as np


def mean_cov(X):
    """Function  calculates the mean and covariance of a data set"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.zeros((1, X.shape[1]))
    mean[0] = np.mean(X, axis=0)
    C = np.dot(X.T, X - mean) / (X.shape[0] - 1)
    return (mean, C)
