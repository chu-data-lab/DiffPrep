import torch
import torch.nn as nn
from copy import deepcopy
from torch.distributions.utils import logits_to_probs, probs_to_logits
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from TFs.mv_imputer import NumMVIdentity, CatMVIdentity
from sklearn.preprocessing import MinMaxScaler
import time

def sinkhorn(X, eps=1e-6, max_iter=500):
    X = torch.exp(X)
    for i in range(max_iter):
        col_sum = X.sum(axis=1, keepdim=True)
        X = X / col_sum
        row_sum = X.sum(axis=2, keepdim=True)
        X = X / row_sum

        if ((col_sum - 1).abs() < eps).all() and ((row_sum - 1).abs() < eps).all():
            return X
    # print("Warning: Sinkhorn not converge")
    return X


class Transformer(nn.Module):
    """ One transformer in data preparation pipeline

    Params:
        tf_options (list): list of transformation functions.
    """
    def __init__(self, name, tf_options, in_features, init_tf=(None, None), diff_method="num_diff", beta=None):
        super(Transformer, self).__init__()
        self.name = name
        self.tf_options = deepcopy(tf_options)
        self.tf_methods = [tf.method for tf in self.tf_options]
        self.num_tf_options = len(tf_options)
        self.diff_method = diff_method
        self.in_features = in_features
        self.out_features = in_features # assume dim not changed
        self.init_tf_option, self.init_p = init_tf
        self.init_parameters(beta)

    def init_parameters(self, beta):
        # probs of executing each tf in every transformer
        # the default is set to 0.5, others are 0.5 / (num_tf - 1)
        self.tf_prob_logits = beta
        # samples
        self.tf_prob_sample = None #shape (num_features, num_tfs)
        self.is_sampled = False

    def numerical_diff(self, X, alpha, eps=1e-6):
        X = X.detach().numpy()
        X_pos = X + eps
        X_neg = X - eps

        X_grads = []
        for tf in self.tf_options:
            f1 = tf.transform(X_pos)
            f2 = tf.transform(X_neg)
            grad = (f1 - f2) / (2 * eps)
            X_grads.append(np.expand_dims(grad, axis=-1))

        X_grads = np.concatenate(X_grads, axis=2)
        X_sample_grad = (X_grads * self.tf_prob_sample.detach().numpy()).sum(axis=2) * alpha.detach().numpy()
        X_sample_grad = np.clip(X_sample_grad, -10, 10)
        return torch.Tensor(X_sample_grad)

    def forward(self, X, is_fit, X_type, max_only=False, require_grad=True, alpha=None):
        # train tfs
        X_trans = []
        for tf in self.tf_options:
            if is_fit:
                X_t = tf.fit_transform(X.detach().numpy()) # transformer sample change, X_output change
            else:
                X_t = tf.transform(X.detach().numpy())

            X_t = torch.Tensor(X_t).unsqueeze(-1)
            X_trans.append(X_t)

        # All transformations
        X_trans = torch.cat(X_trans, dim=2) # shape (num_examples, num_features, num_tfs)

        # select the sample from X transformations
        X_output = self.select_X_sample(X, X_trans, max_only, require_grad=require_grad, alpha=alpha)
        return X_output

    def select_X_sample(self, X, X_trans, max_only, require_grad=True, alpha=None):
        if max_only:
            # transformer_prob_sample, tf_prob_sample = self.sample_with_max_probs()
            tf_prob_sample = self.sample_with_max_probs()
        else:
            # transformer_prob_sample, tf_prob_sample = self.transformer_prob_sample, self.tf_prob_sample
            tf_prob_sample = self.tf_prob_sample

        # print(tf_prob_sample)
        alpha = alpha.unsqueeze(0)
        X_trans_sample = (X_trans * tf_prob_sample.unsqueeze(0)).sum(axis=2) * alpha

        if not require_grad:
            return X_trans_sample
            
        # compute grad
        if self.diff_method == "num_diff":
            X_grad = self.numerical_diff(X, alpha)
        else:
            raise Exception("invalid diff method {}".format(self.diff_method))

        # print(self.name, X_grad.max())
        X_output = X_trans_sample + X_grad * X - (X_grad * X).detach()

        return X_output

    def categorical_max(self, logits):
        max_idx = torch.argmax(logits, dim=1)
        max_sample = torch.zeros_like(logits)
        max_sample[np.arange(max_sample.shape[0]), max_idx] = 1
        return max_sample

    def categorical_sample(self, logits, temperature, use_sample=True):
        # if self.name == "feature_selection":
        #     print(logits_to_probs(logits))
        if not use_sample:
            samples = logits_to_probs(logits, is_binary=False)
        else:
            # print(self.name)
            # print(logits)
            samples = torch.distributions.RelaxedOneHotCategorical(temperature, logits=logits).rsample()
            indicator = torch.max(samples, dim=-1, keepdim=True)[1]  # find one-hot position, an int
            one_h = torch.zeros_like(samples).scatter_(-1, indicator, 1.0)  # change [0,...,0] to one-hot
            diff = one_h - samples.detach()
            samples = samples + diff  # one-hot like variable, can backward
        return samples

    def sample(self, temperature, use_sample=True):
        self.tf_prob_sample = self.categorical_sample(self.tf_prob_logits, temperature, use_sample)
        self.is_sampled = True

    def sample_with_max_probs(self):
        tf_prob_sample = self.categorical_max(self.tf_prob_logits)
        return tf_prob_sample

    def show_alpha(self):
        print("tf logits", self.tf_prob_logits.data)

    def show_probs(self):
        print("tf probs", logits_to_probs(self.tf_prob_logits.data))

class FirstTransformer(Transformer):
    """" The first transformer in the pipeline. Cleaning missing values and one-hot encoding

    Params:
        tf_options: missing value imputers
    """
    def __init__(self, num_tf_options, cat_tf_options, init_num_tf=(None, None), init_cat_tf=(None, None)):
        super(Transformer, self).__init__()
        self.name = "missing_value_imputation"
        self.num_tf_options = num_tf_options
        self.cat_tf_options = cat_tf_options
        self.num_tf_methods = [tf.method for tf in self.num_tf_options]
        self.cat_tf_methods = [tf.method for tf in self.cat_tf_options]
        self.num_num_tf_options = len(num_tf_options)
        self.num_cat_tf_options = len(cat_tf_options)
        self.init_num_tf_option, self.init_num_p = init_num_tf
        self.init_cat_tf_option, self.init_cat_p = init_cat_tf
        self.cache = {}

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

        self.out_num_features = 0
        self.out_cat_features = 0
        self.cache["train"] = {"X_num_trans":None, "X_cat_trans":None}

        if self.contain_num:
            for tf in self.num_tf_options:
                assert tf.input_type in ["numerical", "mixed"]
                if tf.input_type == "numerical":
                    X_num_t = tf.fit_transform(X_num.values)
                    X_num_trans.append(X_num_t)
                else:
                    X_num_t, X_cat_t = tf.fit_transform(X_num.values, X_cat.values)
                    X_num_trans.append(X_num_t)

            X_num_trans = torch.Tensor(np.array(X_num_trans)).permute(1, 2, 0) # shape (num_examples, num_features, num_tfs)
            self.cache["train"]["X_num_trans"] = X_num_trans
            # save numerical column indices (place num before cat)
            self.out_num_features = X_num_trans.shape[1]

        if self.contain_cat:
            for tf in self.cat_tf_options:
                assert tf.input_type in ["categorical", "mixed"]
                if tf.input_type == "categorical":
                    X_cat_t = tf.fit_transform(X_cat.values)
                    X_cat_trans.append(X_cat_t)
                else:
                    X_num_t, X_cat_t = tf.fit_transform(X_num.values, X_cat.values)
                    X_cat_trans.append(X_cat_t)

            # fit one hot encoder on all results of X_cat
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_cat_trans_concat = np.vstack(X_cat_trans)
            self.one_hot_encoder.fit(X_cat_trans_concat)
            X_cat_trans_enc = []
            for X_cat_t in X_cat_trans:
                X_cat_trans_enc.append(self.one_hot_encoder.transform(X_cat_t))

            X_cat_trans = torch.Tensor(np.array(X_cat_trans_enc)).permute(1, 2, 0) # shape (num_examples, num_features, num_tfs)
            self.cache["train"]["X_cat_trans"] = X_cat_trans
            self.out_cat_features = X_cat_trans.shape[1]

        self.out_features = self.out_num_features + self.out_cat_features
        self.feature_names = self.get_feature_names()
        self.init_parameters()

    def get_feature_names(self):
        num_feature_names = [c for c in self.num_columns]
        if self.contain_cat:
            cat_feature_names = [c for c in self.one_hot_encoder.get_feature_names_out(self.cat_columns)]
        else:
            cat_feature_names = []

        feature_names = num_feature_names + cat_feature_names
        return feature_names

    def init_parameters(self):
        self.num_tf_prob_logits = None
        self.cat_tf_prob_logits = None
        self.num_tf_prob_sample = None
        self.cat_tf_prob_sample = None

        if self.contain_num:
            if self.init_num_tf_option is None:
                num_tf_prob_logits = torch.randn(self.out_num_features, self.num_num_tf_options)
            else:
                init_num_tf_idx = self.num_tf_methods.index(self.init_num_tf_option.method)
                init_num_tf_probs = torch.ones(self.out_num_features, self.num_num_tf_options) * ((1-self.init_num_p) / (self.num_num_tf_options - 1))
                init_num_tf_probs[:, init_num_tf_idx] = self.init_num_p
                num_tf_prob_logits = probs_to_logits(init_num_tf_probs)

            self.num_tf_prob_logits = nn.Parameter(num_tf_prob_logits, requires_grad=True) # [feature_num, tf_num]

        if self.contain_cat:
            if self.init_cat_tf_option is None:
                cat_tf_prob_logits = torch.randn(self.out_cat_features, self.num_cat_tf_options) # [feature_num, tf_num]
            else:
                init_cat_tf_idx = self.cat_tf_methods.index(self.init_cat_tf_option.method)
                init_cat_tf_probs = torch.ones(self.out_cat_features, self.num_cat_tf_options) * ((1-self.init_cat_p) / (self.num_cat_tf_options - 1))
                init_cat_tf_probs[:, init_cat_tf_idx] = self.init_cat_p
                cat_tf_prob_logits = probs_to_logits(init_cat_tf_probs)

            self.cat_tf_prob_logits = nn.Parameter(cat_tf_prob_logits, requires_grad=True) # [feature_num, tf_num]

    def transform(self, X):
        X_num = X[self.num_columns]
        X_cat = X[self.cat_columns]
        X_num_trans = []
        X_cat_trans = []

        if self.contain_num:
            for tf in self.num_tf_options:
                if tf.input_type == "numerical":
                    X_num_t = tf.transform(X_num.values)
                    X_num_trans.append(X_num_t)
                else:
                    X_num_t, X_cat_t = tf.transform(X_num.values, X_cat.values)
                    X_num_trans.append(X_num_t)

            X_num_trans = torch.Tensor(np.array(X_num_trans)).permute(1, 2, 0) # shape (num_examples, num_features, num_tfs)

        if self.contain_cat:
            for tf in self.cat_tf_options:
                if tf.input_type == "categorical":
                    X_cat_t = tf.transform(X_cat.values)
                    X_cat_t = self.one_hot_encoder.transform(X_cat_t)
                    X_cat_trans.append(X_cat_t)
                else:
                    X_num_t, X_cat_t = tf.transform(X_num.values, X_cat.values)
                    X_cat_t = self.one_hot_encoder.transform(X_cat_t)
                    X_cat_trans.append(X_cat_t)

            X_cat_trans = torch.Tensor(np.array(X_cat_trans)).permute(1, 2, 0) # shape (num_examples, num_features, num_tfs)

        return X_num_trans, X_cat_trans

    def forward(self, X, is_fit, X_type, max_only=False, require_grad=True):
        """ Forward pass
        Params:
            X (pd.DataFrame): numerical and categorical columns with missing values
        """
        indices = X.index
        if self.contain_num:
            X_num_trans = self.cache[X_type]["X_num_trans"][indices]
        else:
            X_num_trans = None

        if self.contain_cat:
            X_cat_trans = self.cache[X_type]["X_cat_trans"][indices]
        else:
            X_cat_trans = None

        # select the sample from X transformations
        X_output = self.select_X_sample(X_num_trans, X_cat_trans, max_only)
        return X_output

    def pre_cache(self, X, X_type):
        X_num_trans, X_cat_trans = self.transform(X)
        self.cache[X_type] = {
            "X_num_trans": X_num_trans,
            "X_cat_trans": X_cat_trans
        }

    def select_X_sample(self, X_num_trans, X_cat_trans, max_only):
        if max_only:
            num_tf_prob_sample, cat_tf_prob_sample = self.sample_with_max_probs()
        else:
            num_tf_prob_sample, cat_tf_prob_sample = self.num_tf_prob_sample, self.cat_tf_prob_sample

        X_num_trans_sample = None
        X_cat_trans_sample = None

        if num_tf_prob_sample is not None:
            X_num_trans_sample = (X_num_trans * num_tf_prob_sample.unsqueeze(0)).sum(axis=2)

        if cat_tf_prob_sample is not None:
            X_cat_trans_sample = (X_cat_trans * cat_tf_prob_sample.unsqueeze(0)).sum(axis=2)

        X_output = self.concat_num_cat(X_num_trans_sample, X_cat_trans_sample)
        return X_output

    def sample_with_max_probs(self):
        num_tf_prob_sample = None
        cat_tf_prob_sample = None
        if self.num_tf_prob_logits is not None:
            num_tf_prob_sample = self.categorical_max(self.num_tf_prob_logits)

        if self.cat_tf_prob_logits is not None:
            cat_tf_prob_sample = self.categorical_max(self.cat_tf_prob_logits)
        return num_tf_prob_sample, cat_tf_prob_sample

    def sample(self, temperature=0.1, use_sample=True):
        if self.num_tf_prob_logits is not None:
            self.num_tf_prob_sample = self.categorical_sample(self.num_tf_prob_logits, temperature, use_sample)

        if self.cat_tf_prob_logits is not None:
            self.cat_tf_prob_sample = self.categorical_sample(self.cat_tf_prob_logits, temperature, use_sample)

        self.is_sampled = True

    def concat_num_cat(self, X_num, X_cat):
        if X_num is None:
            return X_cat
        if X_cat is None:
            return X_num
        X_output = torch.cat((X_num, X_cat), dim=1)
        return X_output

def is_contain_mv(df):
    return df.isnull().values.sum() > 0

class DiffPrepFlexPipeline(nn.Module):
    """ Data preparation pipeline"""
    def __init__(self, prep_space, temperature=0.1, use_sample=False, diff_method="num_diff", init_method="default"):
        super(DiffPrepFlexPipeline, self).__init__()
        self.prep_space = prep_space
        self.temperature = temperature
        self.use_sample = use_sample
        self.diff_method = diff_method
        self.is_fitted = False
        self.init_method = init_method
        self.n_tf_types = len(prep_space)

    def init_parameters(self, X_train, X_val, X_test):
        pipeline = []
        self.contain_mv = is_contain_mv(X_train) or is_contain_mv(X_val) or is_contain_mv(X_test)

        if self.contain_mv:
            first_tf_dict = self.prep_space[0]

            if self.init_method == "default":
                init_num_tf = first_tf_dict["init"][0]
                init_cat_tf = first_tf_dict["init"][1]
            elif self.init_method == "random":
                init_num_tf = (None, None)
                init_cat_tf = (None, None)
            else:
                raise Exception("Wrong init method")

            first_transformer = FirstTransformer(first_tf_dict["num_tf_options"],
                                                 first_tf_dict["cat_tf_options"],
                                                 init_num_tf = init_num_tf,
                                                 init_cat_tf = init_cat_tf)
        else:
            first_transformer = FirstTransformer([NumMVIdentity()], [CatMVIdentity()])

        first_transformer.fit_transform(X_train)
        first_transformer.pre_cache(X_val, "val")
        first_transformer.pre_cache(X_test, "test")

        pipeline.append(first_transformer)

        # other transformers
        in_features = first_transformer.out_features

        beta_list = []
        for tf_dict in self.prep_space[1:]:

            if self.init_method == "default":
                init_tf = tf_dict["init"]
            elif self.init_method == "random":
                init_tf = (None, None)
            else:
                raise Exception("Wrong init method")
            tf_options = deepcopy(tf_dict["tf_options"])
            tf_methods = [tf.method for tf in tf_options]
            num_tf_options = len(tf_dict["tf_options"])
            init_tf_option, init_p = init_tf
            if init_tf_option is None:
                tf_prob_logits = torch.randn(in_features, num_tf_options)
            else:
                init_tf_probs = torch.ones(in_features, num_tf_options) * (
                        (1 - init_p) / (num_tf_options - 1))
                init_idx = tf_methods.index(init_tf_option.method)
                init_tf_probs[:, init_idx] = init_p
                tf_prob_logits = probs_to_logits(init_tf_probs)
            # ****
            tf_prob_logits = nn.Parameter(tf_prob_logits, requires_grad=True)
            beta_list.append(tf_prob_logits)

        for i in range(self.n_tf_types - 1):
            idx = 0
            for tf_dict in self.prep_space[1:]:

                if self.init_method == "default":
                    init_tf = tf_dict["init"]
                elif self.init_method == "random":
                    init_tf = (None, None)
                else:
                    raise Exception("Wrong init method")
                tf_prob_logits = beta_list[idx]
                transformer = Transformer(tf_dict["name"], tf_dict["tf_options"], in_features,
                                          init_tf=init_tf, diff_method=self.diff_method, beta=tf_prob_logits)
                pipeline.append(transformer)
                idx += 1
        
        self.pipeline = nn.ModuleList(pipeline)
        # self.pipeline = nn.ModuleList([pipeline[i] for i in [0, 1, 5, 9]])
        self.out_features = in_features

        # parameters
        self.alpha = torch.nn.Parameter(torch.randn(in_features, self.n_tf_types-1, self.n_tf_types-1), requires_grad=True)
        # self.alpha = torch.nn.Parameter(torch.eye(self.n_tf_types-1) * 10, requires_grad=True)
        self.alpha_probs = sinkhorn(self.alpha)


    def forward(self, X, is_fit, X_type, resample=False, max_only=False, require_grad=True):
        X_output = deepcopy(X)
        # do first step
        transformer = self.pipeline[0]
        # print('doing ', transformer.name)
        if resample or not transformer.is_sampled:
            transformer.sample(temperature=self.temperature, use_sample=self.use_sample)
        X_output = transformer(X_output, is_fit, X_type, max_only=max_only)

        # do later steps

        self.alpha_probs = sinkhorn(self.alpha)
        # print(self.alpha_probs[0])
        # print(self.alpha_probs[1])
        # self.alpha_probs = torch.eye(3)
        
        cur_idx = 1
        for i in range(self.n_tf_types - 1):
            X_output_i = torch.zeros_like(X_output)

            for j in range(self.n_tf_types - 1):
                transformer = self.pipeline[cur_idx]

                # print('doing ', transformer.name)

                if resample or not transformer.is_sampled:
                    # if i == 0:
                    transformer.sample(temperature=self.temperature, use_sample=self.use_sample)

                # forward
                X_output_i += transformer(X_output, is_fit, X_type, max_only=max_only, require_grad=require_grad, alpha=self.alpha_probs[:, i, j])
                cur_idx += 1
                
                # print(X_output_i.max())

            X_output = X_output_i
        
        return X_output

    def fit(self, X):
        self.is_fitted = True
        return self.forward(X, is_fit=True, X_type="train", resample=True)

    def transform(self, X, X_type, max_only=False, resample=False, require_grad=True):
        "max_only: only do transformer with prob > 0.5 and tf with maximum prob"

        if not self.is_fitted:
            raise Exception("transformer is not fitted")
        return self.forward(X, is_fit=False, X_type=X_type, resample=resample, max_only=max_only, require_grad=require_grad)

    def get_final_dataset(self, X, X_type):
        X_output = self.forward(X, is_fit=False, X_type=X_type, resample=False, max_only=True)
        return X_output

