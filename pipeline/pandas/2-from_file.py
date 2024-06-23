#!/usr/bin/env python3
'''
    function def from_file(filename, delimiter):
    that loads data from a file as a pd.DataFrame:
'''


import pandas as pd


def from_file(filename, delimiter):
    '''
        Args:
            - filename is the file to load from
            - delimiter is the column separator

        Returns:
            - Returns: the loaded pd.DataFrame
    '''

    return pd.read_csv(filename, delimiter=delimiter)
