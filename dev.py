# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:25:51 2017

@author: misaka-wa
"""
import os
data_root = './comparison/DSSP'
sources = map(lambda path: f'{data_root}/{path}', os.listdir(data_root))
from research.datasets_report import DatasetsReport
from research.plot import plot_frequency

sources = list(sources)[:50]

whole = DatasetsReport(*sources).analyze(filtf=lambda v, std, mean: v>mean+3*std)
number_of_dist = len(whole)
for test_some_case_dist in list(whole.keys()):
    if whole[test_some_case_dist][0][1]>0.4:
        plot_frequency(whole[test_some_case_dist])
    
    
    
