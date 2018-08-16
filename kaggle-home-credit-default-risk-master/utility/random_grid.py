"""
|--------------------------------------------------------------------------
FUNCTION - random_grid

INPUT:
----------
    param_dist : dict
        for each parameter, expect a list of candidate values, or a scipy.stat distribution
    
    random_iter : integer
        number of parameter combinations
        
    random_state : integer
        random numpy seed 
OUTPUT:
----------
    a dataframe containing random combinations of param_dist
|--------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import random

def random_grid(param_dist, random_iter = 10, random_state = 1):
    np.random.seed(seed=random_state)
    param_grid = []
    for i in range(random_iter):
        params = {}
        for key, value in param_dist.items():
            if isinstance(value, list):
                value = random.choice(value)
            elif hasattr(value, "rvs"): # if value is a distribution (attribute rvs to generate a random value)
                value = value.rvs()
            else:
                raise TypeError("Paramter {} is a {}. Only list or scipy.stats distribution is supported.".format(key, type(value)))

            params[key] = value
        param_grid.append(params)

    param_grid = pd.DataFrame(param_grid)
    param_grid = param_grid[list(param_dist.keys())] # keep the column order the same as param_dist
    return param_grid