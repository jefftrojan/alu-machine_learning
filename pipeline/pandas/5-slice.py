#!/usr/bin/env python3
"""
    script to slice the pd.DataFrame
    along the columns High, Low, Close,
    and Volume_BTC, taking every 60th row:
"""


import pandas as pd

from_file = __import__("2-from_file").from_file

df = from_file("../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ",")

df = df.iloc[::60, [2, 3, 4, 5]]

print(df.tail())
