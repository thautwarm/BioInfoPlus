import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, precision_recall_fscore_support


def add_doc(f):
    def apply(method):
        method.__doc__ = f.__doc__
        return method

    return apply


class DataSet:
    def __init__(self, samples: np.ndarray, targets: np.ndarray):
        assert len(samples) == len(targets)
        self.samples = samples
        self.targets = targets
        self.is_discrete = all(
            isinstance(target, (np.float64, np.int64, np.int8, np.int16,
                                np.int32)) for target in targets)

    def __len__(self):
        return len(self.targets)

    @add_doc(train_test_split)
    def split(self, test_size):
        train_test_split(self.samples, self.targets, test_size=test_size)

    @add_doc(KFold)
    def kfold(self, n_split: int, shuffle=False):
        kf = KFold(n_split, shuffle=shuffle)
        for tr, te in kf.split(self.samples):
            yield (self.samples[tr], self.targets[tr]), (self.samples[te],
                                                         self.targets[te])

    @add_doc(cross_val_score)
    def cross_val_score(self, estimator, n_jobs=1, cv=None, verbose=False):
        return cross_val_score(
            estimator,
            self.samples,
            self.targets,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose)

    def cross_validate_with_train(self, estimator_creator, n_split=10):
        def stream():
            for (tr_x, tr_y), (te_x, te_y) in self.kfold(
                    n_split, shuffle=True):
                est = estimator_creator()
                est.fit(tr_x, tr_y)

                yield precision_recall_fscore_support(te_y, est.predict(te_x))

        pre, recall, fscore, support = zip(*stream())
        support = np.sum(support, 0)
        pre = np.mean(pre, 0)
        recall = np.mean(recall, 0)
        fscore = np.mean(fscore, 0)
        return [pre, recall, fscore, support]

    def report_classification(self, estimator, io=print):
        prediction = estimator.predict(self.samples)
        io(classification_report(self.targets, prediction))
