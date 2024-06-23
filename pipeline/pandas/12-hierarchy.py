#!/usr/bin/env python3
'''
    Based on 11-concat.py, rearrange the MultiIndex
    levels such that timestamp is the first level:

    TODO:
    - Concatenate the bitstamp and coinbase tables from timestamps
    1417411980 to 1417417980, inclusive
    - Add keys to the data labeled bitstamp and
    coinbase respectively
    - Display the rows in chronological order

'''

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load data
df1 = from_file('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('../Data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Set 'Timestamp' as index
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# Filter the dataframes by the specified timestamps
df1 = df1.loc[1417411980:1417417980]
df2 = df2.loc[1417411980:1417417980]

# Add keys to the data
df1['key'] = 'coinbase'
df2['key'] = 'bitstamp'

# Reset index to add 'Timestamp' as a column
df1 = df1.reset_index()
df2 = df2.reset_index()

# Set new MultiIndex with 'Timestamp' first and 'key' second
df1 = df1.set_index(['Timestamp', 'key'])
df2 = df2.set_index(['Timestamp', 'key'])

# Concatenate the dataframes
df = pd.concat([df1, df2])

# Sort the DataFrame by the new MultiIndex
df = df.sort_index()

# Print the DataFrame
print(df)
