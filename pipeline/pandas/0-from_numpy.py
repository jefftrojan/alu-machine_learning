#!/usr/bin/env python3
'''
    Function def from_numpy(array):
    that creates a pd.DataFrame from a np.ndarray
'''


import string
import pandas as pd


def from_numpy(array):
    '''
        Function def from_numpy(array):
        that creates a pd.DataFrame from a np.ndarray

        Args:
            - array is the np.ndarray from which you should
            create the pd.DataFrame
            - The columns of the pd.DataFrame should be labeled
            in alphabetical order and capitalized.

        Returns:
            - Returns: the newly created pd.DataFrame
    '''
    num_columns = array.shape[1]

    # Generate the column labels (A, B, C, ...)
    columns = [chr(65 + i) for i in range(num_columns)]

    # Create the DataFrame
    df = pd.DataFrame(array, columns=columns)

    return df
