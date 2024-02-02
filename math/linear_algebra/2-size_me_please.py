#!/usr/bin/env python3
"""matrix shape"""


def matrix_shape(matrix):
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
