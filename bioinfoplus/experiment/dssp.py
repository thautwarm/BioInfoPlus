import pandas as pd
import numpy as np
import random
from bioinfoplus.experiment.solver import Solver, mlp, RegularNet
from bioinfoplus.algorithm.ngram import make_gram
from bioinfoplus.algorithm.encoding import Encoder
from bioinfoplus.algorithm.dist import get_conditional_dist
from collections import Counter
import typing as t

gram__slots = ['secondary', 'primary']

hold = object()


def to_list_seq(seqs):
    return (list(seq) for seq in seqs)


class Gram:
    __slots__ = ['seq']

    def __init__(self, seq=None, secondary=None, primary=None):
        if seq is not None:
            self.seq = seq
        else:
            assert len(primary) == len(secondary)
            self.seq = (*primary, secondary)

    def lens(self, loc: int, primary=hold, secondary=hold):

        seq = list(self.seq)
        if primary is not hold:
            seq[loc] = primary
        if secondary is not hold:
            seq[loc + len(self)] = secondary
        return Gram(seq)

    @property
    def primary(self):
        return self.seq[:len(self)]

    @property
    def secondary(self):
        return self.seq[len(self):]

    def __eq__(self, other):
        return isinstance(other, Gram) and all(
            a == b for a, b in zip(self.seq, other.seq))

    def __hash__(self):
        return hash(tuple(self.seq))

    def __len__(self):
        return len(self.seq) // 2

    def __repr__(self):

        return 'Gram[{}]'.format(', '.join(
            f'(pri={pri!r}, sec={sec!r})'
            for pri, sec in zip(self.primary, self.secondary)))


class Analytic:
    def __init__(self, *raw_dfs: pd.DataFrame, gram_size=6, md=None):
        self.raw_dfs = raw_dfs
        self.gram_size = gram_size

        primary_set = set()
        secondary_set = set()

        for raw_df in self.raw_dfs:
            primary_set.update(raw_df.AA)
            secondary_set.update(raw_df.STRUCTURE)
        self.primary_encoder = primary_encoder = Encoder(primary_set)
        self.secondary_encoder = secondary_encoder = Encoder(secondary_set)

        self.solver = Solver(gram_size, len(primary_set), len(secondary_set))

        self.primary_seqs = primary_seqs = []
        self.secondary_seqs = secondary_seqs = []

        for raw_df in self.raw_dfs:
            amino_acid = primary_encoder.transform(raw_df.AA)
            structure = secondary_encoder.transform(raw_df.STRUCTURE)
            primary_seqs.extend(
                to_list_seq(
                    make_gram(amino_acid, gram_size=gram_size, stride=1)))
            secondary_seqs.extend(
                to_list_seq(
                    make_gram(structure, gram_size=gram_size, stride=1)))

    def encode_data(self, *raw_dfs):
        self.primary_seqs = primary_seqs = []
        self.secondary_seqs = secondary_seqs = []
        primary_encoder = self.primary_encoder
        secondary_encoder = self.secondary_encoder
        gram_size = self.gram_size
        for raw_df in self.raw_dfs:
            amino_acid = primary_encoder.transform(raw_df.AA)
            structure = secondary_encoder.transform(raw_df.STRUCTURE)
            primary_seqs.extend(to_list_seq(make_gram(amino_acid, gram_size=gram_size, stride=1)))
            secondary_seqs.extend(to_list_seq(make_gram(structure, gram_size=gram_size, stride=1)))

        return primary_seqs, secondary_seqs

    def build_solver(self, *, lr=0.0001, epoch=100, verbose=True):
        self.solver.tuning_constraint(
            self.primary_seqs,
            self.secondary_seqs,
            lr=lr,
            epoch=epoch,
            verbose=verbose)

    def resolve(self,
                primary_inputs,
                secondary_inputs,
                *,
                lr=0.0001,
                epoch=100,
                verbose=True):
        resolve = self.solver.resolve_variable

        def stream():
            for primary_input, secondary_input in zip(primary_inputs,
                                                      secondary_inputs):
                yield resolve(
                    primary_input,
                    secondary_input)

        return zip(*stream())

    def ml_test(self, test_loc: int,
                grams: t.Union[np.array, pd.DataFrame, t.List[Gram]], md):
        if isinstance(grams, pd.DataFrame):
            test_data = self.encode_data(grams)
        elif isinstance(grams, np.ndarray):
            test_data = grams
        else:
            encoder = self.encoder
            test_data = np.array(
                [encoder.transform(each.seq) for each in grams])
        idx = self.gram_size + test_loc

        def split(xy):
            target_data = xy[:, idx]
            sample_data = xy[:, np.arange(2 * self.gram_size) != idx]
            return sample_data, target_data

        train_data = self.data
        md.fit(*split(train_data))
        print(md.score(*split(test_data)))

    def query(self, seq: t.Union[Gram, t.List[t.Tuple[str, str]]]):
        raise NotImplemented
