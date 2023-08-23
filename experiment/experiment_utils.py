import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import os
import json
import shutil
from itertools import product
from copy import deepcopy
import utils
from sklearn.preprocessing import MinMaxScaler
import time

def makedir(dir_list, file=None, remove_old_dir=False):
    save_dir = os.path.join(*dir_list)

    if remove_old_dir and os.path.exists(save_dir) and file is None:
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir

def load_df(file_path, dataset_info):
    df = pd.read_csv(file_path)

    if "drop_variables" in dataset_info.keys():
        df = df.drop(columns=dataset_info["drop_variables"])

    if 'categorical_variables' in dataset_info.keys():
        categories = dataset_info['categorical_variables']
        for cat in categories:
            df[cat] = df[cat].astype(str).replace('nan', np.nan)
    return df

def split(X, y, val_ratio=0.2, test_ratio=0.2, random_state=1):
    np.random.seed(random_state)
    N = len(y)
    
    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)
    n_train = N - n_test - n_val

    indices = np.random.permutation(N)
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test+n_val]
    train_indices = indices[n_test+n_val:n_test+n_val+n_train]
    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_val = X.iloc[val_indices]
    y_val = y[val_indices]
    X_test = X.iloc[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_val, y_val, X_test, y_test

def remove_all_na(X_train, X_val, X_test):
    is_all_na = X_train.isnull().values.all(axis=0)
    columns = X_train.columns[~is_all_na]
    X_train = X_train[columns]
    X_val = X_val[columns]
    X_test = X_test[columns]
    return X_train, X_val, X_test

def remove_large_cat(X):
    # remove columns with over 1000 of categories
    large_cat_columns = []
    cat_columns = X.select_dtypes(exclude='number').columns
    for c in cat_columns:
        n_cat = len(set(X[c].dropna().values))
        if n_cat > 1000:
            large_cat_columns.append(c)
    columns = [c for c in X.columns if c not in large_cat_columns]
    X = X[columns]
    return X

def build_data(X, y, random_state=1):
    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y.values.ravel())
    y_enc = torch.tensor(y_enc).long()
    
    X = remove_large_cat(X)
    # print("Data size:", X.shape)

    X_train, y_train, X_val, y_val, X_test, y_test = split(X, y_enc, random_state=random_state)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # remove columns with all mvs in training
    X_train, X_val, X_test = remove_all_na(X_train, X_val, X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test

def min_max_normalize(X_train, X_val, X_test):
    num_columns = X_train.select_dtypes(include='number').columns
    cat_columns = X_train.select_dtypes(exclude='number').columns

    X_train_num = X_train[num_columns]
    X_val_num = X_val[num_columns]
    X_test_num = X_test[num_columns]
    X_train_cat = X_train[cat_columns]
    X_val_cat = X_val[cat_columns]
    X_test_cat = X_test[cat_columns]

    scaler = MinMaxScaler()
    X_train_num_norm = pd.DataFrame(scaler.fit_transform(X_train_num.values), columns=num_columns)
    X_val_num_norm = pd.DataFrame(scaler.transform(X_val_num.values), columns=num_columns)
    X_test_num_norm = pd.DataFrame(scaler.transform(X_test_num.values), columns=num_columns)

    X_train = pd.concat([X_train_num_norm, X_train_cat], axis=1)[X_train.columns]
    X_val = pd.concat([X_val_num_norm, X_val_cat], axis=1)[X_train.columns]
    X_test = pd.concat([X_test_num_norm, X_test_cat], axis=1)[X_train.columns]
    return X_train, X_val, X_test

def load_info(info_dir):
    info_path = os.path.join(info_dir, "info.json")
    with open(info_path) as info_data:
        info = json.load(info_data)
    return info

def load_data(data_dir, dataset):
    # load info dict
    dataset_dir = os.path.join(data_dir, dataset)
    info = load_info(dataset_dir)

    file_path = os.path.join(dataset_dir, "data.csv")
    data = load_df(file_path, info)

    label_column = info["label"]
    feature_column = [c for c in data.columns if c != label_column]
    X = data[feature_column]
    y = data[[label_column]]
    return X, y

def set_random_seed(params):
    random_state = params["train_seed"]
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if "cuda" in params["device"]:
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

def save_result(result, model_dict, logger, params, save_dir, save_model=False):
    # save logger
    if logger is not None:
        logger.save(utils.makedir([save_dir, "logging"]))

    # save params and results
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)

    # save model
    if save_model and model_dict is not None:
        for name, model in model_dict.items():
            torch.save(model, os.path.join(save_dir, "{}.pth".format(name)))

def copy_result(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def grid_search(experiment_obj, param_grid, verbose=True):
    # best acc with early stopping
    best_result = None
    best_model = None
    best_logger = None
    best_val_loss = float("inf")
    best_params = None
    
    for i, params in enumerate(get_param_candidates(param_grid)):
        if verbose:
            print("Model lr {}".format(params["model_lr"]))
        
        if "no_crash" in params and params["no_crash"]:
            try:
                result, model, logger = experiment_obj.run(params, verbose=verbose)
            except:
                print("Error!!!!!!")
                continue
        else:
            result, model, logger = experiment_obj.run(params, verbose=verbose)

        if result["best_val_loss"] < best_val_loss:
            best_val_loss = result["best_val_loss"]
            best_result = result
            best_model = model
            best_logger = logger
            best_params = params

    return best_result, best_model, best_logger, best_params

def get_param_candidates(param_grid):
    fixed_params = {}
    tuned_params = {}
    candidate_params = []

    for name, parameter in param_grid.items():
        if type(parameter) == list:
            tuned_params[name] = parameter
        else:
            fixed_params[name] = parameter

    for tuned_params_cand in product(*tuned_params.values()):
        param_cand = deepcopy(fixed_params)
        for n, p in zip(tuned_params.keys(), tuned_params_cand):
            param_cand[n] = p
        candidate_params.append(param_cand)

    return candidate_params

def print_params(model):
    for name, w in model.named_parameters():
        print(name, w.shape)