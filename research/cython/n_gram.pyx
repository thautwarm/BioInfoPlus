# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:37:15 2017

@author: misaka-wa
"""
import numpy as np

def make_gram(numpy_arr:np.ndarray, N:int, step = 1):
    row = numpy_arr.shape[0]
    return np.array([numpy_arr[i:i+N] for i in range(0, row-N+1, step)])

def make_kernel_gram(numpy_arr:np.ndarray, N:int, step = 1, kernel = lambda x : x):
    row = numpy_arr.shape[0]
    return np.array([kernel(numpy_arr[i:i+N]) for i in range(0, row-N+1, step)])

