#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (float): cost of the network without L2 regularization.
        lambtha (float): regularization parameter.
        weights (dict): weights and biases of the network.
        L (int): number of layers in the neural network.
        m (int): number of data points used.

    Returns:
        float: cost of the network accounting for L2 regularization.
    """
    W_sum = 0
    for i in range(1, L + 1):
        W_sum += np.linalg.norm(weights['W' + str(i)])
    return cost + (lambtha / (2 * m)) * W_sum
