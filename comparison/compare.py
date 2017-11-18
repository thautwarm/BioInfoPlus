# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:06:20 2017

@author: misakawa
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics.classification import classification_report
import numpy as np


def optional(f, default):
    try:
        return f()
    except:
        return default


def encode_sequence(origin_X: np.ndarray):
    get_labels = lambda data: data[:, data.shape[1] // 2]
    origin_y = get_labels(origin_X)
    feature_size = origin_X.shape[1]
    y_species = list(np.unique(origin_y))
    X_species = [list(np.unique(field)) for field in origin_X.T]

    def transform(seq, experience: list):
        return np.array([optional(lambda: experience.index(elem), -1) for elem in seq])

    def transformer(X):
        assert X.shape[1] == feature_size
        y = transform(get_labels(X), y_species)
        X = np.array(
            [[optional(lambda: transform(elem, X_species[i]), -1) for elem in field] for i, field in enumerate(X.T)])
        return X, y

    return transformer


class Experiment:
    def __init__(self, datasets, clf_type=RandomForestClassifier):
        self.clf_type = clf_type
        self.origin_datasets = datasets
        self.datasets = None
        self.transformer = encode_sequence(np.array(datasets))

    def compute(self, k=10):
        self.datasets = (X, y) = self.transformer(self.origin_datasets)
        kf = KFold(n_splits=10, shuffle=True)
        self.clfs = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = self.clf_type()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
