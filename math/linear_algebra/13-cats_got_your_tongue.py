#!/usr/bin/env python3
"""
Cats got your tongue 
"""


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    import numpy as np
    return np.concatenate((mat1, mat2), axis)
