import pandas as pd
import numpy as np
from bioinfoplus.algorithm.encoding import Encoder
from bioinfoplus.algorithm.ngram import make_gram
from bioinfoplus.utils.data.sets import DataSet
from sklearn.decomposition import PCA

AA = [
    '!', 'G', 'Z', 'D', 'P', 'K', 'I', 'Y', 'J', 'C', 'M', 'V', 'E', 'A', 'W',
    'T', 'N', 'R', 'O', 'U', 'B', 'Q', 'F', '!*', 'S', 'X', 'L', 'H'
]
GramAA = sum([[(a, b) for b in range(len(AA))] for a in range(len(AA))], [])
print(GramAA)
S = ['', 'T', 'I', 'X', 'S', 'E', 'G', 'H', 'B']


def gram2(seq):
    return [(a, b) for a, b in zip(seq[:-1], seq[1:])]


class Analytic:
    def __init__(self, *raw_dfs: pd.DataFrame, seq_length=7, use_2gram=True):
        assert seq_length % 2
        self.md = None
        self.seq_length = seq_length
        self.use_2gram = use_2gram
        self.pca = PCA()

        self.primary_encoder = primary_encoder = Encoder(AA)
        self.secondary_encoder = secondary_encoder = Encoder(S)
        self.primary_gram_encoder = gram_encoder = Encoder(GramAA)

        primary_seqs = []
        secondary_seqs = []

        mid = seq_length // 2

        def get_center(item):
            return item[mid]

        if use_2gram:
            for raw_df in raw_dfs:
                amino_acid = primary_encoder.transform(raw_df.AA)

                structure = secondary_encoder.transform(raw_df.STRUCTURE)

                for sample, target in zip(
                        make_gram(amino_acid, gram_size=seq_length, stride=1),
                        map(
                            get_center,
                            make_gram(
                                structure, gram_size=seq_length, stride=1))):
                    sample = np.append(sample,
                                       [gram_encoder.transform(gram2(sample))])
                    primary_seqs.append(sample)
                    secondary_seqs.append(target)
        else:
            for raw_df in raw_dfs:
                amino_acid = primary_encoder.transform(raw_df.AA)
                structure = secondary_encoder.transform(raw_df.STRUCTURE)
                primary_seqs.extend(
                    make_gram(amino_acid, gram_size=seq_length, stride=1))
                secondary_seqs.extend(
                    map(get_center,
                        make_gram(structure, gram_size=seq_length, stride=1)))

        self.data = DataSet(self.pca.fit_transform(np.array(primary_seqs)), np.array(secondary_seqs))

    def encode_data(self, *raw_dfs):
        seq_length = self.seq_length
        gram_encoder = self.primary_gram_encoder
        mid = seq_length // 2
        use_2gram = self.use_2gram
        primary_encoder = self.primary_encoder
        secondary_encoder = self.secondary_encoder

        def get_center(item):
            return item[mid]

        primary_seqs = []
        secondary_seqs = []

        if use_2gram:
            for raw_df in raw_dfs:
                amino_acid = primary_encoder.transform(raw_df.AA)

                structure = secondary_encoder.transform(raw_df.STRUCTURE)

                for sample, target in zip(
                        make_gram(amino_acid, gram_size=seq_length, stride=1),
                        map(
                            get_center,
                            make_gram(
                                structure, gram_size=seq_length, stride=1))):
                    sample = np.append(sample,
                                       [gram_encoder.transform(gram2(sample))])
                    primary_seqs.append(sample)
                    secondary_seqs.append(target)
        else:
            for raw_df in raw_dfs:
                amino_acid = primary_encoder.transform(raw_df.AA)
                structure = secondary_encoder.transform(raw_df.STRUCTURE)
                primary_seqs.extend(
                    make_gram(amino_acid, gram_size=seq_length, stride=1))
                secondary_seqs.extend(
                    map(get_center,
                        make_gram(structure, gram_size=seq_length, stride=1)))

        return DataSet(self.pca.fit_transform(np.vstack(primary_seqs)), np.array(secondary_seqs))
