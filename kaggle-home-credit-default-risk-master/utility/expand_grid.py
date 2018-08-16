"""
|--------------------------------------------------------------------------
FUNCTION - expand_grid

INPUT:
----------
    data_dict : dict
    
OUTPUT:
----------
    a dataframe containing all combinations of data_dict
|--------------------------------------------------------------------------
"""
import pandas as pd
import itertools

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())