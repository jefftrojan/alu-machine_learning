#!/usr/bin/env python3

"""
This module contains a function that calculates
probability density function of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    initializes variables for a Gaussian Mixture Model

    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    m: numpy.ndarray (d,) mean of the distribution
    S: numpy.ndarray (d, d) covariance matrix of the distribution

    return:
        - P: numpy.ndarray (n,) the PDF values for each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or d != S.shape[0] or d != S.shape[1]:
        return None
    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    fac = 1 / np.sqrt(((2 * np.pi) ** d) * S_det)
    X_m = X - m
    X_m_dot = np.dot(X_m, S_inv)
    X_m_dot_X_m = np.sum(X_m_dot * X_m, axis=1)
    P = fac * np.exp(-0.5 * X_m_dot_X_m)
    return np.maximum(P, 1e-300)
