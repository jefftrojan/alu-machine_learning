#!/usr/bin/env python3
"""
7 getting cozy from math/linear_algebra
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None
    if axis == 0:
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    return [mat1[i] + mat2[i] for i in range(len(mat1))]
