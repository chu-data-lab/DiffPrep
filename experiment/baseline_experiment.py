import utils
from .experiment_utils import build_data
from model import LogisticRegression
from trainer.baseline_trainer import BaselineTrainerSGD
import torch
import torch.nn as nn
from .experiment_utils import set_random_seed, load_data, makedir, grid_search, save_result
from pipeline.baseline_pipeline import BaselinePipeline
from tqdm import tqdm
import time
import os
from utils import SummaryWriter
import pandas as pd

class BaselineExperiment(object):
    """Run baseline with one set of hyper parameters"""
    def __init__(self, data_dir, dataset, prep_space, method, model_name, tf_seed=1):
        self.data_dir = data_dir
        self.dataset = dataset
        self.prep_space = prep_space
        self.method = method
        self.model_name = model_name
        self.tf_seed = tf_seed
        
    def run(self, params, verbose=True):
        # load data
        X, y = load_data(self.data_dir, self.dataset)
        X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=params["split_seed"])
        
        # data prep pipeline
        prep_pipeline = BaselinePipeline(self.method, self.prep_space, self.tf_seed)
        X_train = prep_pipeline.fit_transform(X_train, X_val, X_test)
        X_val = prep_pipeline.transform(X_val)
        X_test = prep_pipeline.transform(X_test)

        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        X_test = torch.Tensor(X_test)

        # model
        input_dim = X_train.shape[1]
        output_dim = len(set(y.values.ravel()))
        
        # set experiment seed
        set_random_seed(params)
        if self.model_name == "log":
            model = LogisticRegression(input_dim, output_dim)
        else:
            raise Exception("Wrong model name")

        # if self.method == "equinn_norm":
        #     model = EquiNNNorm(input_dim, self.prep_space, model)
            
        model = model.to(params["device"])

        # loss
        loss_fn = nn.CrossEntropyLoss()

        # optimizer
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["model_lr"]
        )

        model_scheduler = None

        if params["logging"]:
            logger = SummaryWriter()
            logger.add_baseline_pipeline(prep_pipeline.pipeline, global_step=0)
        else:
            logger = None

        baseline = BaselineTrainerSGD(model, loss_fn, model_optimizer, model_scheduler, params, writer=logger)
        result, model = baseline.fit(X_train, y_train, X_val, y_val, X_test, y_test, verbose=verbose)

        return result, model, logger

def random_search(data_dir, dataset, result_dir, prep_space, params, model_name, method, num_random):
    best_val_loss = float("inf")
    best_result = None
    
    summary = []
    for seed in tqdm(range(num_random)):
        baseline_random = BaselineExperiment(data_dir, dataset, prep_space, method, model_name, tf_seed=seed)
        result, model, logger, params_i = grid_search(baseline_random, params, verbose=False)
        result["tf_seed"] = seed
        # print(result["best_val_acc"])

        if result["best_val_loss"] < best_val_loss:
            best_val_loss = result["best_val_loss"]
            best_result = result
            best_model = model
            best_logger = logger
            best_params = params_i

        if seed+1 in [20, 50]:
            print(dataset, "random {}".format(seed+1))
            print("val acc:", best_result["best_val_acc"], "test acc", best_result["best_test_acc"])
            save_dir = makedir([result_dir, "random_{}".format(seed + 1)], remove_old_dir=True)
            save_result(best_result, best_model, best_logger, best_params, save_dir, save_model=False)
        
        summary.append(result)
        pd.DataFrame(summary).to_csv(makedir([result_dir], "summary.csv"), index=False)

def run_baseline(data_dir, dataset, result_dir, prep_space, params, model_name, method, num_random=20):
    if method == "default":
        # baseline 1: default
        baseline_default = BaselineExperiment(data_dir, dataset, prep_space, "default", model_name)
        default_result, default_model, default_logger, default_params = grid_search(baseline_default, params)
        save_result(default_result, default_model, default_logger, default_params, result_dir, save_model=False)
        print("val acc:", default_result["best_val_acc"], "test acc", default_result["best_test_acc"])

    elif method == "random":
        # baseline 2: random_fix
        random_search(data_dir, dataset, result_dir, prep_space, params, model_name, method, num_random)
