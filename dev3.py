# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 01:16:25 2018

@author: twshe
"""

from Redy.Tools.PathLib import Path
from bioinfoplus.utils.dssp.parser import parse
from bioinfoplus.utils.dssp.preprocess import preprocess
from bioinfoplus.experiment.classic import Analytic

from bioinfoplus.algorithm.gather_tree import GatherTree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
dfs = []

paths = Path(
    r'C:\Users\twshe\git\tex\BioInfoPlus\downloader\data').list_dir()

for each in paths[:100]:
    dfs.append(preprocess(parse(str(each))))


analytic = Analytic(*dfs, seq_length=7, use_2gram=False)
a, *_ = analytic.data.cross_validate_with_train(RandomForestClassifier)
print(np.mean(a))

analytic = Analytic(*dfs, seq_length=7, use_2gram=True)
a, *_ = analytic.data.cross_validate_with_train(RandomForestClassifier)
print(np.mean(a))