# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:18:54 2017

@author: misakawa
"""
import warnings
try:
    import pyximport
    pyximport.install()
    from .cython.datasets_report import *
    
except:
    warnings.warn('''C-extension failed, use original python. 
              Download VS2017 C++ Build Tools to speed algorithms!''')
    from .original.datasets_report import *
#def _Analyze(grams: np.ndarray):
#    indices = np.arange(grams.shape[1])
#    
#    ranges = [np.array([0])]
#    dist_indices = [[(0, 'A'), (1, 'B')]]
#    stats  = [[('A',0.1)]]
#    
#    ranges.pop(); stats.pop();dist_indices.pop()
#    
#    for col_idx in indices:
#        ranges.append(np.unique(grams[:, col_idx]))
#    
#    for length in indices:
#        for location_permutation in permutations(indices, int(length+1)):
#            for case_permutation in gen_permutations([ranges[i] for i in location_permutation]):
#                case_frequency = specific_report(grams, [location_permutation, case_permutation])
#                _freq = list(case_frequency.values())
#                std, mean = np.std(_freq), np.mean(_freq)
#                inf = mean - std
#                sup = mean + std
#                case_frequency = {key:value for key, value in case_frequency.items() if not (inf<value<sup)}
#                if case_frequency:
#                    dist_indices.append(case_permutation)
#                    stats.append(case_frequency)
#    
#    return ranges, dist_indices, stats


    
                
                
            
                    
                
                
    
    
    
    
                
        
        
        
    
