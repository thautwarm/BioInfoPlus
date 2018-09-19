# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:27:52 2017

@author: misakawa
"""

from bioinfoplus.utils.dssp.parser import parse
from research.n_gram import make_gram
from research.whole_regular import whole_report
import numpy as np

class DatasetsReport:
    def __init__(self, *paths:'Paths of your datasets', number_of_gram = 5):
        grams = []
        for path in paths:
            dataframe = parse(path)
            AA = dataframe.AA.map(lambda x:x.replace(' ', ''))
            Structure = dataframe.STRUCTURE 
            cases = np.array([AA, Structure]).T
            grams +=  [each_gram.T.flatten() for each_gram in make_gram(cases, number_of_gram)]
        self.grams = np.array(grams)
        self.ranges = None
    
    def analyze(self, filtf:'how to filter some cases'=None):
        """filtf(v:'porbability of this case', std, mean) -> bool
        """
        return {k: v for k, (v, is_to_saved) in 
                        ((k, task_for_case(v, filtf=filtf)) for k,v in whole_report(self.grams).items()) 
                     if is_to_saved}

def task_for_case(case_dist, filtf=None):
    N = sum(case_dist.values())
    case_dist = {k:v/N for k,v in case_dist.items()}
    values = list(case_dist.values())
    std, mean = np.std(values), np.mean(values)
    inf, sup = mean-std, mean+std
    
    if filtf is None:
        _filtf = lambda v: not (inf<v<sup) 
    else:
        _filtf = lambda v: filtf(v, std, mean)
        
    ret = sorted([(k, v) for k,v in case_dist.items()], key=lambda x:-x[1])
    
    return ret, _filtf(ret[0][1]) if ret else False