import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from missingpy import MissForest
from impyute.imputation.cs import em
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

class EMImputer(object):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        X_trans = em(X)
        return  X_trans

class ModeImputer(object):
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='most_frequent')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X_num, X_cat):
        if X_num.shape[1] > 0:
            self.num_imputer.fit(X_num)
        else:
            self.num_imputer = None
        if X_cat.shape[1] > 0:
            self.cat_imputer.fit(X_cat)
        else:
            self.cat_imputer = None

    def transform(self, X_num, X_cat):
        X_num_trans = None
        X_cat_trans = None
        if self.num_imputer is not None:
            X_num_trans = self.num_imputer.transform(X_num)
        if self.cat_imputer is not None:
            X_cat_trans = self.cat_imputer.transform(X_cat)
        return X_num_trans, X_cat_trans

class NumMVImputer(object):
    """Impute missing values on numerical columns. Inputs must be all numerical columns and missing values
    are filled with np.nan

    Available mv imputation methods:
        - 'mean': impute missing values with mean. 
        - 'median': impute missing values with median.
        - 'EM': imputation based on EM. 
        - 'MICE': imputation based on Multivariate Imputation by Chained Equations.
        - 'KNN': k-nearest neighbor imputation (k=5).
    """
    def __init__(self, method='mean'):
        self.input_type = "numerical"
        self.method = method
        if self.method == "mean":
            self.tf = SimpleImputer(strategy='mean')
        elif self.method == "median":
            self.tf = SimpleImputer(strategy='median')
        elif self.method == "EM":
            self.tf = EMImputer()
        elif self.method == "KNN":
            self.tf = KNNImputer(n_neighbors=5)
        elif self.method == "MICE":
            self.tf = IterativeImputer(random_state=0, skip_complete=True)
        elif self.method == "DT":
            self.tf = IterativeImputer(DecisionTreeRegressor(max_features='sqrt', random_state=0), random_state=0, skip_complete=True)
        else:
            raise Exception("Invalid normalization method: {}".format(method))

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        X = self.tf.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans

class CatMVImputer(object):
    """Impute missing values on categorical columns

    Available mv imputation methods:
        - 'dummy': add a new category called 'dummy_category'
    """
    def __init__(self, method='dummy'):
        self.input_type = "categorical"
        self.method = method
        if self.method == "dummy":
            self.tf = SimpleImputer(strategy="constant", fill_value="dummy_category")
        else:
            raise Exception("Invalid normalization method: {}".format(method))

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        X = self.tf.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        X_trans = self.transform(X)
        return X_trans

class NumCatMVImputer(object):
    """Impute missing values on both numerical and categorical columns

    Available mv imputation methods:
        - 'mode': impute missing values with the most frequent value.
        - 'MF': missing value imputation based on MissForest
        - 'mean_mode': impute num with mean and cat with the most frequent value.
    """
    def __init__(self, method='mode'):
        self.input_type = "mixed"
        self.method = method
        if self.method == "mode":
            self.tf = ModeImputer()
        # TODO
        # elif self.method == "MF":
        #     self.tf = MissForestImputer()
        else:
            raise Exception("Invalid normalization method: {}".format(method))

    def fit(self, X_num, X_cat):
        self.tf.fit(X_num, X_cat)

    def transform(self, X_num, X_cat):
        X_num, X_cat = self.tf.transform(X_num, X_cat)
        return X_num, X_cat

    def fit_transform(self, X_num, X_cat):
        self.fit(X_num, X_cat)
        X_num_trans, X_cat_trans = self.transform(X_num, X_cat)
        return X_num_trans, X_cat_trans

class NumMVIdentity(object):
    """Identity Transformer"""
    def __init__(self):
        self.method = "num_mv_identity"
        self.input_type = "numerical"
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X

class CatMVIdentity(object):
    """Identity Transformer"""
    def __init__(self):
        self.method = "cat_mv_identity"
        self.input_type = "categorical"
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X

""" Test """
# X = pd.read_csv("data.csv")
# X_num = X.select_dtypes(include='number')
# X_cat = X.select_dtypes(exclude='number')

# imputer = NumCatMVImputer("MF")
# X_num_repair, X_cat_repair = imputer.fit_transform(X_num.values, X_cat.values)
# X_num_repair = pd.DataFrame(X_num_repair, columns=X_num.columns)
# X_cat_repair = pd.DataFrame(X_cat_repair, columns=X_cat.columns)
# X_num_repair.to_csv("X_num_repair.csv", index=False)
# X_cat_repair.to_csv("X_cat_repair.csv", index=False)

# imputer = NumMVImputer("EM")
# X_num_repair = imputer.fit_transform(X_num.values)
# X_num_repair = pd.DataFrame(X_num_repair, columns=X_num.columns)
# X_num_repair.to_csv("X_num_repair.csv", index=False)



