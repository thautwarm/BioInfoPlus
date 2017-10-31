# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:36:38 2017

@author: misaka-wa
"""



import numpy as np
from collections import defaultdict
from typing import Dict, Union, Any

def specific_report(numpy_arr:np.ndarray, specific:Union[np.ndarray, Dict[int, Any]]):
    """
    numpy_arr : 2D np.array
    specific  : [np.array, np.array]
    """
    if isinstance(specific, dict):
        specific = list(zip(*(specific.items())))
        index    = np.array(specific[0])
        value    = np.array(specific[1])
    else:    
        index:int = specific[0]
        value:int = specific[1]
        
    switch : Dict[int]  = defaultdict(int)
    for case in numpy_arr:
        if  all(case[index] == value):
            switch[tuple(case)] += 1
    summary = sum([i for i in switch.values()])
    return {key:value/summary for key, value in switch.items()}


