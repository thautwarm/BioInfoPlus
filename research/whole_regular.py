# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:16:48 2017

@author: misakawa
"""

import warnings
try:
    import pyximport
    pyximport.install()
    from .cython.whole_regular import *
    
except:
    warnings.warn('''C-extension failed, use original python. 
              Download VS2017 C++ Build Tools to speed algorithms!''')
    from .original.whole_regular import *