import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class UniformDiscretizer(object):
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, X):
        self.x_max = X.max(axis=0, keepdims=True)
        self.x_min = X.min(axis=0, keepdims=True)
        self.step = (self.x_max - self.x_min) / self.n_bins

    def transform(self, X):
        X_trans = (X - self.x_min) / (self.step + 1e-12)
        X_trans = X_trans.astype(int)
        X_trans = np.clip(X_trans, 0, self.n_bins-1)
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans

class QuantileDiscretizer(object):
    def __init__(self, n_bins):
        self.n_bins = n_bins

    def fit(self, X):
        sort_X = np.sort(X, axis=0)
        step = round(len(X) / self.n_bins)
        indices = np.arange(step-1, len(X)-1, step)
        self.split = sort_X[indices, :]

    def transform(self, X):
        X_trans = np.zeros_like(X)

        for i in range(self.split.shape[0]):
            X_trans += X > self.split[i:i+1]

        X_trans = np.clip(X_trans, 0, self.n_bins-1)
        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans

class Discretizer(object):
    """Discretize data

    Args:
        n_bins: number of bins
        strategy: {'uniform', 'quantile', 'kmeans'}
    """
    def __init__(self, n_bins=5, strategy='uniform'):
        self.method = "{}_{}".format(strategy, n_bins)
        if strategy == "uniform":
            self.tf = UniformDiscretizer(n_bins)
        else:
            self.tf = QuantileDiscretizer(n_bins)

    def fit(self, X):
        self.tf.fit(X)
        raise

    def transform(self, X):
        return self.tf.transform(X)

    def fit_transform(self, X):
        # tic = time.time()
        X_out = self.tf.fit_transform(X)
        # print("XXXX", self.method, time.time() - tic)
        return X_out
