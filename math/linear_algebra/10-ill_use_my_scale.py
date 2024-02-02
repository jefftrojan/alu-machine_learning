#!/usr/bin/env python3
"""
Calculate the shape of an Ndarray
"""


def np_shape(matrix):
    """
    Calculate the shape of an Ndarray
    """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape + [len(matrix)]

