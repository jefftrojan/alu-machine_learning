#!/usr/bin/env python3
"""
Bracing the elements
"""


def np_elementwise(mat1, mat2):
    """
    Adds, subtracts, multiplies, and divides two matrices element-wise
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
