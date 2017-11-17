# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:18:14 2017

@author: misakawa

Making statistical analysis on some dssp datas.

"""


from preprocess.BioParser import bio_parse
from research.n_gram import make_gram
from research.specific_regular import specific_report
import numpy as np
from matplotlib import pyplot as plt

# 读数据文件
sources = ['./dssp/sources/1a00.dssp', 
           './dssp/sources/1a0a.dssp',
           './dssp/sources/1a0b.dssp',
           './dssp/sources/1a0c.dssp',
           './dssp/sources/1a0d.dssp']
grams = []
for src in sources:
    dataframe = bio_parse(src)
    AA = dataframe.AA.map(lambda x:x.replace(' ', ''))
    Structure = dataframe.STRUCTURE 
    cases = np.array([AA, Structure]).T
    grams +=  [each_gram.T.flatten() for each_gram in make_gram(cases, 5)]  # 5-gram


frequency = specific_report(grams, {0:'V', 1:'A'}) # 研究分布

print(frequency.values())
key2ind = dict(zip(frequency.keys(), range(len(frequency))))
ind =  np.array(list(key2ind.values()), dtype=np.int32)
values = np.array([frequency[key] for key in key2ind])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('First of sequence is V, while the second is A.')
ax.set_ylabel('Probability')
ax.bar(ind, values)
ax.set_xticks(ind)
x_labeled = ax.set_xticklabels([f'c{i}' for i in ind])
plt.show()


# 对多个数据集做分析

from research.datasets_report import DatasetsReport
from research.plot import plot_frequency
whole = DatasetsReport(*sources).analyze(filtf=lambda probability, std, mean: probability>0.4)
number_of_dist = len(whole)

for test_some_case_dist in list(whole.keys()):
    plot_frequency(whole[test_some_case_dist])





