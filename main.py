import utils
import argparse
import time
from prep_space import space
from experiment.baseline_experiment import run_baseline
from experiment.diffprep_experiment import run_diffprep
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--data_dir', default="data")
parser.add_argument('--result_dir', default="result")
parser.add_argument('--model', default="log", choices=["log", "two", "reg"])
parser.add_argument('--method',  default="diffprep_fix", choices=["default", "random", "diffprep_fix", "diffprep_flex"])
parser.add_argument('--train_seed', default=1, type=int)
parser.add_argument('--split_seed', default=1, type=int)
parser.add_argument('--task', default="classification", choices=["classification", "regression"])
args = parser.parse_args()

# define hyper parameters
params = {
    "num_epochs": 2000,
    "batch_size": 512,
    "device": "cuda",
    "model_lr": [0.1, 0.001, 0.0001],
    # "model_lr": [0.01],
    "weight_decay": 0,
    "model": args.model,
    "train_seed": args.train_seed,
    "split_seed": args.split_seed,
    "method": args.method,
    "save_model": True,
    "logging": False,
    "no_crash": False,
    "patience": 3,
    "momentum": 0.9,
    "task": args.task
}

auto_prep_params = {
    "prep_lr": None,
    "temperature": 0.1,
    "grad_clip": None,
    "pipeline_update_sample_size": 512,
    "init_method": "default",
    "diff_method": "num_diff",
    "sample": False
}

params.update(auto_prep_params)

if args.dataset is None:
    datasets = sorted(os.listdir(args.data_dir))
else:
    datasets = [args.dataset]

for i, dataset in enumerate(datasets):
    print("Run {} on dataset {}".format(args.method, dataset))

    result_dir = utils.makedir([args.result_dir, args.method, dataset])

    if args.method in ["diffprep_fix", "diffprep_flex"]:
        run_diffprep(args.data_dir, dataset, result_dir, space, params, args.model, args.method)
    else:
        run_baseline(args.data_dir, dataset, result_dir, space, params, args.model, args.method)