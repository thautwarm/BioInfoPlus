# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:25:51 2017

@author: misaka-wa
"""

# <Test> [preprocessing::BioParser]
from preprocess.BioParser import bio_parse
dataframe = bio_parse('./dssp/sources/1a00.dssp')
AA = dataframe.AA # amino acid
Structure = dataframe.STRUCTURE # secondary structure

# <Test> [research::cython::n_gram], [research::n_gram]
from research.cython.n_gram import make_gram as cmg
from research.n_gram import make_gram as mg
import numpy as np
x = np.array([1,2,3,4,5])
cmg(x, 2)
mg(x, 2)