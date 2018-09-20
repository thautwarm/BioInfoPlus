from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import safe_sparse_dot
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import normalize
from collections import defaultdict
from bioinfoplus.utils.data.sets import DataSet
import numpy as np


def vote(manys, weights):
    def sort_key(it):
        return it[1]

    voting = defaultdict(int)
    for many in manys:
        for weight, res in zip(weights, many):
            voting[res] += weight
        yield sorted(voting.items(), key=sort_key, reverse=True)[0][0]
        voting.clear()


class GatherTree:
    def __init__(self,
                 min_size=10,
                 max_size=100,
                 initial_score=100,
                 max_warning_times=3,
                 decay=0.89):
        self._initial_score = initial_score
        self._decay = decay
        self._min_size = min_size
        self._max_size = max_size

        self._estimators = np.array(
            [DecisionTreeClassifier() for _ in range(min_size)])
        self._scores = np.zeros(min_size)
        self._scores.fill(min_size)
        self._warnings = np.zeros(min_size)
        self._max_warning_times = max_warning_times

    def fit(self, X, y):
        return self.train(DataSet(X, y))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def train(self, ds: DataSet):
        estimators = self._estimators
        current_size = len(estimators)
        corrs = np.zeros((current_size, current_size))
        for (tr_samples, tr_targets), (te_samples, te_targets) in ds.kfold(
                5, shuffle=True):

            def stream():
                raise NotImplemented

            scores, preds = zip(*stream())
            n = len(preds)
            corr = np.eye(n)
            rows = normalize(preds)

            for i in range(n):
                row_i = rows[i].T
                score_i = scores[i]
                for j in range(i, n):
                    corr[j, i] = corr[i, j] = score_i * scores[j] / (
                        1 + safe_sparse_dot(rows[j], row_i, dense_output=True))

            corrs += corr
        print(corrs)
        scores = self._scores = self._decay * self._scores + corrs.sum(axis=0)
        median = np.median(scores)
        warnings = self._warnings
        warnings[scores < median] += 1
        warnings[scores >= median] -= 1
        warnings.clip(0, self._max_warning_times + 1, out=warnings)
        dropables = warnings > self._max_warning_times
        if any(dropables):
            reserved = np.arange(current_size)[~dropables]
            print(reserved)
            estimators = estimators[reserved]
            scores = scores[reserved]
            warnings = warnings[reserved]
            if len(warnings) < self._min_size:
                increase_num = self._max_size - len(warnings)
                estimators = np.append(
                    estimators,
                    [DecisionTreeClassifier() for _ in range(increase_num)])
                scores = np.append(
                    scores, np.repeat(self._initial_score, increase_num))
                warnings = np.append(warnings, np.zeros(increase_num))

            self._estimators = estimators
            self._scores = scores
            self._warnings = warnings

    def predict(self, samples):
        weight = self._scores
        return np.array(
            tuple(
                vote(
                    zip(*[est.predict(samples) for est in self._estimators]),
                    weight)))
