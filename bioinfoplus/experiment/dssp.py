import pandas as pd
import numpy as np
import sklearn as skl
from bioinfoplus.algorithm.ngram import make_gram
from bioinfoplus.algorithm.encoding import Encoder
from bioinfoplus.algorithm.dist import get_conditional_dist
from collections import Counter
import typing as t

gram__slots = ['secondary', 'primary']

hold = object()


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
    def __init__(self, *raw_dfs: pd.DataFrame, gram_size=6):
        self.raw_dfs = raw_dfs
        self.gram_size = gram_size

        mapping_set = set()
        for raw_df in self.raw_dfs:
            mapping_set.update({*raw_df.AA, *raw_df.STRUCTURE})
        gram_size = self.gram_size
        self.encoder = encoder = Encoder(mapping_set)
        grams = []

        for raw_df in self.raw_dfs:
            structure = encoder.transform(raw_df.STRUCTURE)
            amino_acid = encoder.transform(raw_df.AA)
            data_pairs = np.array(tuple(zip(amino_acid, structure)))
            grams.extend(make_gram(data_pairs, gram_size=gram_size, stride=1))

        self.grams = grams
        self.data = np.array([each_gram.T.flatten() for each_gram in grams])

        # prospective
        self.knowledge_graph = None
        self.ml_md = None

    def encode_data(self, raw_df):
        encoder = self.encoder
        structure = encoder.transform(raw_df.STRUCTURE)
        amino_acid = encoder.transform(raw_df.AA)
        data_pairs = np.array(tuple(zip(amino_acid, structure)))
        grams = make_gram(data_pairs, gram_size=self.gram_size, stride=1)
        return np.array([each_gram.T.flatten() for each_gram in grams])

    def as_fixed_graph(self):
        if not self.knowledge_graph or not self.encoder:
            data = self.data
            knowledge_graph = get_conditional_dist(data)
            for precondition, distribution in knowledge_graph.items():
                distribution_count = sum(distribution.values())
                if distribution_count:
                    for each in tuple(distribution):
                        distribution[each] /= distribution_count
            self.knowledge_graph = knowledge_graph

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
        if isinstance(seq, Gram):
            seq = tuple(zip(seq.primary, seq.secondary))

        if self.knowledge_graph:
            # FLATTEN
            vec = sum(zip(*self.encoder.transform_batch(seq)), ())
            # noinspection PyComparisonWithNone
            prospective: dict = self.knowledge_graph[vec]
            encoded = sorted(
                prospective.items(), key=lambda it: it[1], reverse=True)
            return tuple((Gram(self.encoder.decode(k)), v) for k, v in encoded)
        else:

            def match(gram1, gram2):
                def eq(a, b):
                    return a is None or b is None or a == b

                return all(
                    eq(g1[0], g2[0]) and eq(g1[1], g2[1])
                    for g1, g2 in zip(gram1, gram2))

            def stream():
                for raw_df in self.raw_dfs:
                    structure = raw_df.STRUCTURE
                    amino_acid = raw_df.AA
                    data_pairs = np.array(tuple(zip(amino_acid, structure)))
                    grams = make_gram(
                        data_pairs, gram_size=self.gram_size, stride=1)
                    for each in grams:
                        if match(each, seq):
                            yield sum((zip(*each)), ())

            count = Counter(stream())
            summary = sum(count.values())
            return tuple(
                (Gram(k), v / summary)
                for k, v in sorted(count.items(), key=lambda it: it[1]))


def dssp_to_grams(*raw_dfs, gram_size=6):
    for raw_df in raw_dfs:
        structure = raw_df.STRUCTURE
        amino_acid = raw_df.AA
        data_pairs = np.array(tuple(zip(amino_acid, structure)))
        grams = make_gram(data_pairs, gram_size=gram_size, stride=1)
        yield from map(Gram, [each.T.flatten() for each in grams])
