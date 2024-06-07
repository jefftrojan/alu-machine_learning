#!/usr/bin/env python3
""" Normalize"""


def normalize(X, m, s):
    """ Normalize

    Args:
        X (np.array): with shape (m, nx) to normalize
        m (_type_): _description_
        s (_type_): _description_
    """
    return (X - m) / s
