# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:43:29 2017

@author: misakawa
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_frequency(frequency):
    ind2key = dict(zip(range(len(frequency)), (i[0] for i in frequency)))
    ind =  np.arange(len(frequency))
    bar_split = ind[-1] if ind[-1] < 7 else 7
    values = np.array([i[1] for i in frequency])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('distribution probability')
    ax.set_ylabel('Probability')
    ax.set_xticks(np.arange(0, ind[-1], bar_split))
    ax.bar(ind, values)
    plt.show()
    return ind2key
