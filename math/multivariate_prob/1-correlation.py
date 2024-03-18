#!/usr/bin/env python3
"""Function that calculates a correlation matrix"""

import numpy as np


def correlation(C):
    """Function calculates a correlation matrix"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    v = np.diag(1 / np.sqrt(np.diag(C)))
    CR = np.matmul(np.matmul(v, C), v)
    return CR
