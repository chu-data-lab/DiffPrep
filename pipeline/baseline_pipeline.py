from copy import deepcopy
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from TFs.mv_imputer import NumMVIdentity, CatMVIdentity
import time
class Transformer(object):
    """ One transformer in data preparation pipeline"""
    def __init__(self, name, tf_options, in_features):
        """If thre are more than one tf options, random select one for each feature"""
        self.name = name
        self.tf_options = tf_options
        self.num_options = len(tf_options)

        if len(tf_options) == 1:
            tf_idx_choices = [0 for _ in range(in_features)]
        else:
            tf_idx_choices = np.random.randint(0, self.num_options, size=in_features).tolist()
        
        self.tf_probs = np.zeros((len(tf_idx_choices), self.num_options)) # feature * num_options
        self.tf_probs[range(len(tf_idx_choices)), tf_idx_choices] = 1 # (features,)

    def forward(self, X, is_fit):
        # train tfs
        X_trans = []
        for tf in self.tf_options:
            if is_fit:
                X_t = tf.fit_transform(X) # transformer sample change, X_output change
            else:
                X_t = tf.transform(X)

            X_t = np.expand_dims(X_t, axis=-1)
            X_trans.append(X_t)
        
        # All transformations
        X_trans = np.concatenate(X_trans, axis=2) # shape (num_examples, num_features, num_tfs)
        X_output = np.sum(X_trans * np.expand_dims(self.tf_probs, axis=0), axis=2)
        return X_output

class BaselineFirstTransformer(object):
    """" The first transformer in the pipeline. Cleaning missing values and one-hot encoding

    Params:
        tf_options: missing value imputers
    """
    def __init__(self, num_tf_options, cat_tf_options, in_num_features, in_cat_features, init_num_tf=None, init_cat_tf=None):
        self.name = "missing_value_imputation"
        self.num_tf_options = num_tf_options
        self.cat_tf_options = cat_tf_options
        self.num_tf_probs = None
        self.cat_tf_probs = None

        if init_num_tf is None:
            num_tf_idx_choices = np.random.randint(0, len(num_tf_options), size=in_num_features)
        else:
            num_methods = [tf.method for tf in self.num_tf_options]
            init_num_idx = num_methods.index(init_num_tf.method)
            num_tf_idx_choices = [init_num_idx for _ in range(in_num_features)]

        if init_cat_tf is None:
            cat_tf_idx_choices = np.random.randint(0, len(cat_tf_options), size=in_cat_features)
        else:
            cat_methods = [tf.method for tf in self.cat_tf_options]
            init_cat_idx = cat_methods.index(init_cat_tf.method)
            cat_tf_idx_choices = [init_cat_idx for _ in range(in_cat_features)]

        self.num_tf_idx_choices = num_tf_idx_choices
        self.cat_tf_idx_choices = cat_tf_idx_choices

    def fit_transform(self, X):
        """ Train transformers and Initialize parameters

        Params:
            X (pd.DataFrame): numerical and categorical columns with missing values (np.nan)
        """
        X_num = X.select_dtypes(include='number')
        X_cat = X.select_dtypes(exclude='number')
        self.num_columns = X_num.columns
        self.cat_columns = X_cat.columns

        X_num_trans = []
        X_cat_trans = []
        self.contain_num = X_num.shape[1] > 0
        self.contain_cat = X_cat.shape[1] > 0
        X_num_trans_sample = None
        X_cat_trans_sample = None

        self.out_num_features = 0
        self.out_cat_features = 0

        if self.contain_num:
            for tf in self.num_tf_options:
                # print("num", tf.method)
                if tf.input_type == "numerical":
                    X_num_t = tf.fit_transform(X_num.values)
                    X_num_trans.append(X_num_t)

                if tf.input_type == "mixed":
                    X_num_t, X_cat_t = tf.fit_transform(X_num.values, X_cat.values)
                    X_num_trans.append(X_num_t)

            X_num_trans = np.transpose(np.array(X_num_trans), (1, 2, 0)) # shape (num_examples, num_features, num_tfs)
            self.num_tf_probs = np.zeros((X_num_trans.shape[1], X_num_trans.shape[2])) # num_feature * num_tfs
            self.num_tf_probs[range(len(self.num_tf_idx_choices)), self.num_tf_idx_choices] = 1 # (features,)
            X_num_trans_sample = np.sum(X_num_trans * np.expand_dims(self.num_tf_probs, axis=0), axis=2)
            self.out_num_features = X_num_trans.shape[1]

        if self.contain_cat:
            for tf in self.cat_tf_options:
                # print("cat", tf.method)
                if tf.input_type == "categorical":
                    X_cat_t = tf.fit_transform(X_cat.values)
                    X_cat_trans.append(X_cat_t)

                if tf.input_type == "mixed":
                    X_num_t, X_cat_t = tf.fit_transform(X_num.values, X_cat.values)
                    X_cat_trans.append(X_cat_t)

            # fit one hot encoder on all results of X_cat
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_cat_trans_concat = np.vstack(X_cat_trans)
            self.one_hot_encoder.fit(X_cat_trans_concat)

            X_cat_trans_enc = []
            for X_cat_t in X_cat_trans:
                X_cat_trans_enc.append(self.one_hot_encoder.transform(X_cat_t))
            X_cat_trans = np.transpose(np.array(X_cat_trans_enc), (1, 2, 0)) # shape (num_examples, num_features, num_tfs)

            self.cat_tf_probs = np.zeros((X_cat_trans.shape[1], X_cat_trans.shape[2])) # num_feature * num_options

            # all encoded columns related to the selected original columns have prob 1
            
            idx_map = self.get_feature_enc_map(X_cat.shape[1])
            
            for org_idx, tf_idx in enumerate(self.cat_tf_idx_choices):
                cur_indices = idx_map[org_idx]
                self.cat_tf_probs[cur_indices, tf_idx] = 1

            X_cat_trans_sample = np.sum(X_cat_trans * np.expand_dims(self.cat_tf_probs, axis=0), axis=2)
            self.out_cat_features = X_cat_trans.shape[1]

        self.out_features = self.out_num_features + self.out_cat_features
        X_output = self.concat_num_cat(X_num_trans_sample, X_cat_trans_sample)       
        self.feature_names = self.get_feature_names()
        return X_output

    def get_feature_names(self):
        num_feature_names = [c for c in self.num_columns]
        if self.contain_cat:
            cat_feature_names = [c for c in self.one_hot_encoder.get_feature_names_out(self.cat_columns)]
        else:
            cat_feature_names = []

        feature_names = num_feature_names + cat_feature_names
        return feature_names

    def get_feature_enc_map(self, num_cat_features):
        # original feature idx to feature indices after one hot encoding
        feature_names = self.one_hot_encoder.get_feature_names_out([str(i) for i in range(num_cat_features)])
        idx_map = {}
        for cur_idx, name in enumerate(feature_names):
            org_idx = int(name.split("_")[0])
            if org_idx not in idx_map:
                idx_map[org_idx] = [cur_idx]
            else:
                idx_map[org_idx].append(cur_idx)
        return idx_map

    def forward(self, X, is_fit=False):
        X_num = X[self.num_columns]
        X_cat = X[self.cat_columns]

        X_num_trans = []
        X_cat_trans = []
        X_num_trans_sample = None
        X_cat_trans_sample = None

        if self.contain_num:
            for tf in self.num_tf_options:
                # print(tf.method)
                if tf.input_type == "numerical":
                    X_num_t = tf.transform(X_num.values)
                    X_num_trans.append(X_num_t)

                if tf.input_type == "mixed":
                    X_num_t, X_cat_t = tf.transform(X_num.values, X_cat.values)
                    X_num_trans.append(X_num_t)

            X_num_trans = np.transpose(np.array(X_num_trans), (1, 2, 0)) # shape (num_examples, num_features, num_tfs)
            X_num_trans_sample = np.sum(X_num_trans * np.expand_dims(self.num_tf_probs, axis=0), axis=2)

        if self.contain_cat:
            for tf in self.cat_tf_options:
                if tf.input_type == "categorical":
                    X_cat_t = tf.transform(X_cat.values)
                    X_cat_t = self.one_hot_encoder.transform(X_cat_t)
                    X_cat_trans.append(X_cat_t)

                if tf.input_type == "mixed":
                    X_num_t, X_cat_t = tf.transform(X_num.values, X_cat.values)
                    X_cat_t = self.one_hot_encoder.transform(X_cat_t)
                    X_cat_trans.append(X_cat_t)

            X_cat_trans = np.transpose(np.array(X_cat_trans), (1, 2, 0)) # shape (num_examples, num_features, num_tfs)
            X_cat_trans_sample = np.sum(X_cat_trans * np.expand_dims(self.cat_tf_probs, axis=0), axis=2)

        X_output = self.concat_num_cat(X_num_trans_sample, X_cat_trans_sample)
        return X_output

    def concat_num_cat(self, X_num, X_cat):
        if X_num is None:
            return X_cat
        if X_cat is None:
            return X_num
        X_output = np.concatenate((X_num, X_cat), axis=1)
        return X_output

def is_contain_mv(df):
    return df.isnull().values.sum() > 0

def random_select(X):
    idx = np.random.choice(len(X))
    return X[idx]

def get_num_cat_features(X_train):
    X_num = X_train.select_dtypes(include='number')
    X_cat = X_train.select_dtypes(exclude='number')
    in_num_features = X_num.shape[1]
    in_cat_features = X_cat.shape[1]
    return in_num_features, in_cat_features

class BaselinePipeline(object):
    """ Data preparation pipeline"""
    def __init__(self, method, prep_space, random_state=1):
        assert(method in ["default", "random", "random_flex"])
        self.method = method
        self.prep_space = prep_space
        self.random_state = random_state

    def fit_transform(self, X_train, X_val, X_test):
        np.random.seed(self.random_state)
        self.pipeline = []

        self.contain_mv = is_contain_mv(X_train) or is_contain_mv(X_val) or is_contain_mv(X_test)

        # fit first transformer
        first_tf_dict = self.prep_space[0]

        if not self.contain_mv:
            # no missing value
            num_tf_options = [NumMVIdentity()]
            cat_tf_options = [CatMVIdentity()]

        elif self.method == "default":
            # default 
            num_tf_options = [first_tf_dict["default"][0]]
            cat_tf_options = [first_tf_dict["default"][1]]
           
        elif self.method == "random":
            num_tf_options = [random_select(first_tf_dict["num_tf_options"])]
            cat_tf_options = [random_select(first_tf_dict["cat_tf_options"])]

        else:
            num_tf_options = first_tf_dict["num_tf_options"]
            cat_tf_options = first_tf_dict["cat_tf_options"]

        # transform
        in_num_features, in_cat_features = get_num_cat_features(X_train)
        first_transformer = BaselineFirstTransformer(num_tf_options, cat_tf_options, in_num_features, in_cat_features)
        X_trans = first_transformer.fit_transform(X_train)

        self.pipeline.append(first_transformer)

        for tf_dict in self.prep_space[1:]:
            if self.method == "default":
                tf_options = [tf_dict["default"]]
            elif self.method == "random":
                tf_options = [random_select(tf_dict["tf_options"])]
                # print(tf_options[0].method)
            else:
                tf_options = tf_dict['tf_options']

            transformer = Transformer(tf_dict['name'], tf_options, X_trans.shape[1])
            X_trans = transformer.forward(X_trans, is_fit=True)
            self.pipeline.append(transformer)

        return X_trans

    def transform(self, X):
        X_output = deepcopy(X)

        for transformer in self.pipeline:
            # print('doing ', transformer.name)
            # forward
            X_output = transformer.forward(X_output, is_fit=False)
        return X_output