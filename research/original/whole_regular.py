# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:15:13 2017

@author: misakawa
"""

from collections import defaultdict
from itertools import combinations
import numpy as np
def whole_report(numpy_arr:np.ndarray):
    """
    numpy_arr : 2D np.array
    specific  : [np.array, np.array]
    """
    n_col = numpy_arr.shape[1]
    dims = set(range(n_col))
    switch = defaultdict(lambda : defaultdict(int))
    for case in numpy_arr:
        case = tuple(case)
        for dim_num in range(1, n_col):
            for location_combination in combinations(dims, dim_num):
                location = [None]*n_col
                for loc in location_combination:
                    location[loc] = case[loc]
                location = tuple(location)
                switch[location][case] += 1
    return switch