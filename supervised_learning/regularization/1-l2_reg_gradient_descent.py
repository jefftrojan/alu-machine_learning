#!/usr/bin/env python3
"""
Compute gradient descent with L2 regularization
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Compute gradient descent with L2 regularization

    Args:
        Y (numpy.ndarray): one-hot matrix with the correct labels
        weights (dict): The weights and biases of the network
        cache (dict): The outputs of each layer of the network
        alpha (float): The learning rate
        lambtha (float): The L2 regularization parameter
        L (int): The number of layers of the network
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dw = (1 / m) * np.matmul(dz, A.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(W.T, dz) * (1 - np.square(A))
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db

    return weights
