import numpy as np
import pandas as pd
import utils
from .experiment_utils import set_random_seed, load_data, build_data, grid_search, makedir, save_result
from model import LogisticRegression
from pipeline.diffprep_flex_pipeline import DiffPrepFlexPipeline
from pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
import torch
import torch.nn as nn
from trainer.diffprep_trainer import DiffPrepSGD
from utils import SummaryWriter
from .experiment_utils import min_max_normalize, min_max_y
from copy import deepcopy
from utils import logits_to_probs
import pprint
import os
import json

class DiffPrepExperiment(object):
    """Run auto prep with one set of hyper parameters"""
    def __init__(self, data_dir, dataset, prep_space, model_name, method):
        self.data_dir = data_dir
        self.dataset = dataset
        self.prep_space = prep_space
        self.model_name = model_name
        self.method = method

    def run(self, params, verbose=True):
        X, y = load_data(self.data_dir, self.dataset)
        task = params["task"]
        X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, task, random_state=params["split_seed"])
        
        # pre norm for diffprep flex
        if self.method == "diffprep_flex":
            X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
        if params["task"] == "regression":
            y_train, y_val, y_test = min_max_y(y_train), min_max_y(y_val), min_max_y(y_test)

        params["patience"] = 10
        params["num_epochs"] = 3000
        
        # set random seed
        set_random_seed(params)

        ## transform pipeline
        # define and fit first step
        if self.method == "diffprep_fix":
            prep_pipeline = DiffPrepFixPipeline(self.prep_space, temperature=params["temperature"],
                                             use_sample=params["sample"],
                                             diff_method=params["diff_method"],
                                             init_method=params["init_method"])
        elif self.method == "diffprep_flex":
            prep_pipeline = DiffPrepFlexPipeline(self.prep_space, temperature=params["temperature"],
                            use_sample=params["sample"],
                            diff_method=params["diff_method"],
                            init_method=params["init_method"])
        else:
            raise Exception("Wrong auto prep method")

        prep_pipeline.init_parameters(X_train, X_val, X_test)
        print("Train size: ({}, {})".format(X_train.shape[0], prep_pipeline.out_features))

        # model
        input_dim = prep_pipeline.out_features
        if params["task"] == "classification":
            output_dim = len(set(y.values.ravel()))
        else:
            # As of now only added support for regression
            output_dim = 1

        # model = TwoLayerNet(input_dim, output_dim)
        set_random_seed(params)
        if self.model_name == "log":
            model = LogisticRegression(input_dim, output_dim)
        elif self.model_name == "reg":
            print(input_dim)
            model = torch.nn.Sequential(torch.nn.Linear(input_dim, 4),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(4, 1))
        else:
            raise Exception("Wrong model")

        model = model.to(params["device"])

        # loss
        if params["task"] == "regression":
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        # optimizer
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["model_lr"],
            weight_decay=params["weight_decay"],
            momentum=params["momentum"]
        )
        
        if params["prep_lr"] is None:
            prep_lr = params["model_lr"]
        else:
            prep_lr = params["prep_lr"]
    
        prep_pipeline_optimizer = torch.optim.Adam(
            prep_pipeline.parameters(),
            lr=prep_lr,
            betas=(0.5, 0.999),
            weight_decay=params["weight_decay"]
        )

        # scheduler
        # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=patience, factor=0.1, threshold=0.001)
        prep_pipeline_scheduler = None
        model_scheduler = None

        if params["logging"]:
            logger = SummaryWriter()
        else:
            logger = None

        diff_prep = DiffPrepSGD(prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                    model_scheduler, prep_pipeline_scheduler, params, writer=logger)

        result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, X_test, y_test)
        return result, best_model, logger

def run_diffprep(data_dir, dataset, result_dir, prep_space, params, model_name, method):
    print("Dataset:", dataset, "Task:", params["task"], "Diff Method:", params["diff_method"], method)

    sample = "sample" if params["sample"] else "nosample"
    diff_prep_exp = DiffPrepExperiment(data_dir, dataset, prep_space, model_name, method)
    best_result, best_model, best_logger, best_params = grid_search(diff_prep_exp, deepcopy(params))
    dict = {}
    logits = [dict.update({key: values}) for key, values in best_model['prep_pipeline'].items()]
    # print(logits_to_probs(torch.FloatTensor(logits[0])))
    # print(dict)
    save_lr_and_pipelines(best_params["model_lr"], dict, result_dir)
    # p = pprint.PrettyPrinter(width=41)
    # p.pprint([(key, logits_to_probs(values)) for key, values in best_model['prep_pipeline'].items() if key != 'alpha'])
    # pipe_line = [np.argmax(values, axis=1) for key, values in best_model['prep_pipeline'].items()]
    # print(np.unique(pipe_line[0], return_counts=True, axis=0))
    save_result(best_result, best_model, best_logger, best_params, result_dir, save_model=False)
    print("DiffPrep Finished. val acc:", best_result["best_val_acc"], "test acc", best_result["best_test_acc"])


def logits_to_probs(logits):
    """Convert logits to probabilities."""
    logits = logits.float()  # Convert the tensor to float type
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def tensor_to_list(tensor):
    """Convert tensors to lists for JSON serialization."""
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    return tensor

def save_lr_and_pipelines(lr, pipeline_dict, save_dir):
    # Convert the logits to probabilities and filter out the 'alpha' key
    data_with_probs = [(key, logits_to_probs(value)) for key, value in pipeline_dict.items() if key != 'alpha']

    # Convert the data to a dictionary for saving
    data_to_save_dict = {key: tensor_to_list(value) for key, value in data_with_probs}
    # Decide the filename extension, json is a good format for structured data
    file_path = os.path.join(save_dir, 'bestpipelines.json')
  
    # Select the appropriate keys based on the learning rate
    keys_to_save = ['pipeline.0.num_tf_prob_logits', 'pipeline.0.cat_tf_prob_logits']
    if lr == 0.1:
        keys_to_save += ['pipeline.1.tf_prob_logits', 'pipeline.2.tf_prob_logits', 'pipeline.3.tf_prob_logits']
    elif lr == 0.01:
        keys_to_save += ['pipeline.4.tf_prob_logits', 'pipeline.5.tf_prob_logits', 'pipeline.6.tf_prob_logits']
    elif lr == 0.001:
        keys_to_save += ['pipeline.7.tf_prob_logits', 'pipeline.8.tf_prob_logits', 'pipeline.9.tf_prob_logits']
    
    # print(data_to_save_dict)
    # Extract the relevant data from the dictionary
    data_to_save = {key: data_to_save_dict[key] for key in keys_to_save if key in data_to_save_dict}
    # print(data_to_save)
    # Write the data to a file in the specified directory
    with open(file_path, 'w') as f:
        count = 0
        for key, value in data_to_save.items():
            f.write(key + ": ")
            f.write(str(value))
            f.write("\n")
    # print(f"Data saved to {file_path}")