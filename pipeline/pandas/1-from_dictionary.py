#!/usr/bin/env python3
"""
    python script that created a
    pd.DataFrame from a dictionary:
"""


import pandas as pd


"""
    Function def from_dictionary():
    that creates a pd.DataFrame from a dictionary

    Args:
    - The first column should be labeled First and have the
    values 0.0, 0.5, 1.0, and 1.5
    - The second column should be labeled Second and have the
    values one, two, three, four
    - The rows should be labeled A, B, C, and D, respectively

    Returns:
    - The pd.DataFrame should be saved into the variable df
"""

# Create the dictionary
dictionary = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create the DataFrame
df = pd.DataFrame(dictionary, index=['A', 'B', 'C', 'D'])
