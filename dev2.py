# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 19:59:34 2018

@author: twshe
"""
from bioinfoplus.experiment.solver import *
import torch
pri = [[4, 1, 2, 0, 3], [2, 0, 1, 1, 3]]
sec = [[2, 2, 3, 1, 0], [0, 0, 1, 2, 3]]

pri_s = 5
sec_s = 4

solver = Solver(5, 5, 4)

solver.tuning_constraint(pri, sec, epoch=1000)

print(solver.resolve_variable([4, 1, 2, None, 3], [2, 2, 3, 1, 0], lr=0.01))
