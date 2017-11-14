# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:37:43 2017

@author: misakawa
"""
import warnings
try:
    import pyximport
    pyximport.install()
    from .cython.n_gram import *
    
except:
    warnings.warn('''C-extension failed, use original python. 
              Download VS2017 C++ Build Tools to speed algorithms!''')
    from .original.n_gram import *