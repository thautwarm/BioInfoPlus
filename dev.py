# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:25:51 2017

@author: misaka-wa
"""
from Redy.Tools.PathLib import Path
from bioinfoplus.utils.dssp.parser import parse
from bioinfoplus.utils.dssp.preprocess import preprocess
from bioinfoplus.experiment.dssp import Analytic, dssp_to_grams
dfs = []
test = []
paths = Path(
    r'C:\Users\twshe\git\tex\BioInfoPlus\downloader\data').list_dir()[:500]

print(len(paths))

for each in paths[:-50]:
    dfs.append(preprocess(parse(str(each))))

for each in paths[-50:]:
    test.append(preprocess(parse(str(each))))
analytic = Analytic(*dfs, gram_size=4)

#analytic.as_fixed_graph()
grams = dssp_to_grams(*test, gram_size=4)
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
