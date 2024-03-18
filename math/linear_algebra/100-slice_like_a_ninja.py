#!/usr/bin/env python3
"""
Slice numpy arrays
"""
import numpy as np


def np_slice(matrix, axes=None):
    """Slices a matrix along specific axes

    Args:
        matrix (numpy.ndarray): matrix to slice
        axes (dict): dictionary where the key is an axis to slice along and
                     the value is a tuple representing the slice to make along
                     that axis

    Returns:
        numpy.ndarray: the sliced matrix
    """
    if axes is None:
        return matrix[...]
    slices = []
    for i in range(matrix.ndim):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))
    return matrix[tuple(slices)]
  
mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(np_slice(mat1, axes={1: (1, 3)}))
print(mat1)
# mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
#                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
#                  [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
# print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
# print(mat2)