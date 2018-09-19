from collections import defaultdict, Counter
from itertools import combinations
import numpy as np

_bit_types = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


def get_conditional_dist(numpy_arr: np.ndarray, bit=8):
    """
    numpy_arr : 2D np.array filled with numbers
    """
    _, n_col = numpy_arr.shape
    dims = set(range(n_col))
    switch = defaultdict(Counter)
    location_template = np.zeros(n_col).astype(_bit_types[bit])

    reset = location_template.fill
    reset(-1)
    for case in numpy_arr:
        case = tuple(case)
        for dim_num in range(1, n_col):
            for location_combination in combinations(dims, dim_num):
                for loc in location_combination:
                    location_template[loc] = case[loc]
                location = tuple(location_template)
                reset(-1)
                switch[location][case] += 1
    return switch
