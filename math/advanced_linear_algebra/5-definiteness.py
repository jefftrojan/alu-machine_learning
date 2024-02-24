#!/usr/bin/env python3
"""Function that calculates the definiteness of a matrix"""

import numpy as np


def definiteness(matrix):
    """Function that calculates the definiteness of a matrix"""
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None
    ev, _ = np.linalg.eig(matrix)
    if all(ev < 0):
        return "Negative definite"
    if all(ev <= 0):
        return "Negative semi-definite"
    if all(ev > 0):
        return "Positive definite"
    if all(ev >= 0):
        return "Positive semi-definite"
    else:
        return 'Indefinite'
