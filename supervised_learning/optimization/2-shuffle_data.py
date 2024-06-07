#!/usr/bin/env python3
""" Shuffle Data"""

import numpy as np


def shuffle_data(X, Y):
    """ Shuffle data points in two matrices the same way

    Args:
        X (_type_): _description_
        Y (_type_): _description_
    """
    s = np.random.permutation(X.shape[0])
    return X[s], Y[s]
