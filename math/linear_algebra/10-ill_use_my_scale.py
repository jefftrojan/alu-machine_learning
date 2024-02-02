#!/usr/bin/env python3
"""
Calculate the shape of an Ndarray
"""

import numpy as np
def np_shape(matrix):
    """
    Calculate the shape of an Ndarray
    """
    matrix = np.array(matrix)
    return matrix.shape()

