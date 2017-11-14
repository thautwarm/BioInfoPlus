# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:37:53 2017

@author: misakawa
"""
import warnings
try:
    import pyximport
    pyximport.install()
    from .cython.specific_regular import *
    
except:
    warnings.warn('''C-extension failed, use original python. 
              Download VS2017 C++ Build Tools to speed algorithms!''')
    from .original.specific_regular import *