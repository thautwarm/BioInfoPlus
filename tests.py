# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:25:51 2017

@author: misaka-wa
"""

# <Test> [preprocessing::BioParser]
from preprocess.BioParser import bio_parse
dataframe = bio_parse('./dssp/sources/1a00.dssp')
AA = dataframe.AA.map(lambda x:x.replace(' ', '')) # amino acid
Structure = dataframe.STRUCTURE # secondary structure

# <Test> [research::cython::n_gram], [research::n_gram]
from research.n_gram import make_gram
import numpy as np
x = np.array([1,2,3,4,5])
make_gram(x, 2)


# n_gram
cases = np.array([AA, Structure]).T
res =  [each_gram.T.flatten() for each_gram in make_gram(cases, 5)]
from research.specific_regular import specific_report
frequency = specific_report(res, {1:'A', 2:'A'})
print(frequency.values())

