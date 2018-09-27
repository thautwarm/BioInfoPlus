# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:41:28 2018

@author: twshe
"""

from Redy.Tools.PathLib import Path
from bioinfoplus.utils.dssp.parser import parse
from bioinfoplus.utils.dssp.preprocess import preprocess
from bioinfoplus.experiment.dssp import Analytic
from bioinfoplus.experiment.solver import Solver
from sklearn.model_selection import KFold
import numpy as np

result_file = Path('result').open('w')
dataframes = []
paths = Path(r'downloader\data').list_dir()[:120]
for each in paths:
    dataframes.append(preprocess(parse(str(each))))


def index_seq(seq, indices):
    return [seq[idx] for idx in indices]


def validate(snippet_len,
             primary_species,
             secondary_species,
             primary_seqs,
             secondary_seqs,
             tr_indices,
             te_indices,
             missing_loc='pri:0'):
    kind, offset = missing_loc.split(':')
    offset = int(offset)
    if kind == 'pri':

        def lens(pri, sec):
            pri = list(pri)
            pri[offset] = None
            return pri, sec

    elif kind == 'sec':

        def lens(pri, sec):
            sec = list(sec)
            sec[offset] = None
            return pri, sec
    else:
        raise ValueError

    tr_pri_seqs = index_seq(primary_seqs, tr_indices)

    te_pri_seqs = index_seq(primary_seqs, te_indices)

    tr_sec_seqs = index_seq(secondary_seqs, tr_indices)

    te_sec_seqs = index_seq(secondary_seqs, te_indices)

    solver = Solver(snippet_len, primary_species, secondary_species)

    solver.tuning_constraint(tr_pri_seqs, tr_sec_seqs, epoch=100000, lr=0.001)

    def stream():
        for primary_input, secondary_input in zip(te_pri_seqs, te_sec_seqs):
            primary_input, secondary_input = lens(primary_input,
                                                  secondary_input)
            yield solver.resolve_variable(primary_input, secondary_input)

    def acc_helper(seq):
        return sum(seq) / len(seq)

    pred_pri_seqs, pred_sec_seqs = zip(*stream())

    prediction = tuple(
        pred_pri == te_pri and pred_sec == te_sec
        for pred_pri, pred_sec, te_pri, te_sec in zip(
            pred_pri_seqs, pred_sec_seqs, te_pri_seqs, te_sec_seqs))

    msg = f'| acc {acc_helper(prediction)} snippet_len: {snippet_len} missing_loc {missing_loc}\n'
    result_file.write(msg)
    result_file.flush()


def experiment(
        snippet_range=(4, 10), missing_locs=('sec:1',
                                             'sec:2', 
                                             'pri:1', 
                                             'pri:2')):
    low, high = snippet_range
    for snippet_len in range(low, high):
        analytic = Analytic(*dataframes, gram_size=snippet_len)
        primary_seqs = analytic.primary_seqs
        secondary_seqs = analytic.secondary_seqs
        for tr_indices, te_indices in KFold(
                n_splits=5, shuffle=True).split(primary_seqs):
            for missing_loc in missing_locs:
                validate(snippet_len, analytic.solver.primary_species,
                         analytic.solver.secondary_species, primary_seqs,
                         secondary_seqs, tr_indices, te_indices, missing_loc)


if __name__ == '__main__':
    experiment()
