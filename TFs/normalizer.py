import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
warnings.filterwarnings("ignore")

class Normalizer():
    """Normalize the dataset
    Available methods:
        - 'ZS' z-score normalization
        - 'MM' MinMax scaler
        - 'NQT' Normal QuantileTransformer
        - 'UQT' Uniform QuantileTransformer
        - 'RS' Robust scaling
        - 'MA' MaxAbosobute Scaling
        # - 'DS' decimal scaling
        # - 'Log10' log10 scaling
    """

    def __init__(self, method='ZS'):
        self.method = method
        if self.method == "ZS":
            self.tf = StandardScaler()
        elif self.method == "MM":
            self.tf = MinMaxScaler(clip=True)
        elif self.method == "MA":
            self.tf = MaxAbsScaler()
        elif self.method == "RS":
            self.tf = RobustScaler()
        else:
            raise Exception("Invalid normalization method: {}".format(method))
    
    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        X = self.tf.transform(X)
        X = X.clip(-1e10, 1e10)
        return X

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans