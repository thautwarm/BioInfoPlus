# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:25:51 2017

@author: misaka-wa
"""
from Redy.Tools.PathLib import Path
from bioinfoplus.utils.dssp.parser import parse
from bioinfoplus.utils.dssp.preprocess import preprocess
from bioinfoplus.experiment.dssp import Analytic
import numpy as np
dfs = []
test = []
paths = Path(
    r'C:\Users\twshe\git\tex\BioInfoPlus\downloader\data').list_dir()[:500]

for each in paths[:100]:
    dfs.append(preprocess(parse(str(each))))

for each in paths[100:110]:
    test.append(preprocess(parse(str(each))))
    
print('data ready')
analytic = Analytic(*dfs, gram_size=5)

print('building solver')
analytic.build_solver(lr=0.001, epoch=100000, verbose=True)

print('solver built')

pri, sec = analytic.encode_data(*test)


def lens(seq, loc, value):
    seq = list(seq)
    seq[loc] = value
    return seq

s_pri, s_sec = analytic.resolve([lens(e, loc=2, value=None) for e in pri], 
                                sec)


def stat(seq):
    return seq.sum() / len(seq)
print(stat(np.array([c == d for c, d in zip(s_pri, pri)])))

#    


#analytic.as_fixed_graph()

#for gram in grams:
#    print('================================')
#
#    unified = analytic.query(gram.lens(loc=2, primary=None))
#    print(len(unified))
#    if unified:
#        print(unified[0])
#    else:
#        print('not found')
#    print(gram)
