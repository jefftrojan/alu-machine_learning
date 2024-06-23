#!/usr/bin/env python3
"""
 Calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a neural network with L2 regularization

    Args:
        cost (float): the cost of the network without L2 regularization
        lambtha (_type_):  is the regularization parameter
        weights (_type_): s a dictionary of the weights and biases
        (numpy.ndarrays) of the neural network
        L (int): the number of layers in the neural network
        m (int): the number of data points used
    """
    penality = 0

    for i in range(1, L + 1):
        key = 'W' + str(i)
        penality += np.sum(np.square(weights[key]))

    penality *= (lambtha / (2 * m))

    total_cost = cost + penality

    return total_cost
