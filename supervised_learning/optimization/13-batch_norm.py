#!/usr/bin/env python3
""" Learing rate decay with tensorflow
"""


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a neural network using batch
    normalization

    Args:
        Z (numpy.ndarray): matrix to normalize shape (m, n)
            m: number of data points
            n: number of features
        gamma (numpy.ndarray): shape (1, n)
        contains the scales used for batch normalization
        beta (numpy.ndarray): shape (1, n)
        contains the offsets used for batch normalization
        epsilon (float): small number used to avoid division by zero

    Returns: the normalized Z matrix

    """

    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    Z_norm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Z_tilda = gamma * Z_norm + beta
    return Z_tilda
