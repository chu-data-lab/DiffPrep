import utils
import argparse
from prep_space import space_large as space
from experiment.baseline_experiment import run_baseline
import torch
from datetime import date
import os
import time
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--data_dir', default="data/Dataset0928")
parser.add_argument('--result_dir', default="result")
parser.add_argument('--method', default="random_fix", choices=["default", "random_fix", "random_flex"])
parser.add_argument('--model', default="log", choices=["log", "two"])
parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--split_seed', default=1, type=None)
parser.add_argument('--train_seed', default=1, type=int)
parser.add_argument('--test', action="store_true", default=False)
parser.add_argument('--all', action="store_true", default=False)
parser.add_argument('--no_crash', action="store_true", default=False)
parser.add_argument('--cpu', default=1, type=int)
parser.add_argument('--group', default=0, type=int)
args = parser.parse_args()

# args.all = True

# define hyper parameters
params = {
    "num_epochs": 2000,
    "batch_size": 512,
    "device": "cpu",
    "model_lr": [0.1, 0.01, 0.001],
    "weight_decay": 0,
    "train_seed": args.train_seed,
    "no_crash": args.no_crash,
    "patience": 3,
    "logging": True,
    "split_seed": args.split_seed
}

if args.gpu and torch.cuda.is_available():
    params["device"] = "cuda"

if args.test:
    params["num_epochs"] = 5
    params["model_lr"] = [0.001]

today = date.today().strftime("%m%d")
args.result_dir += today + "_" + args.method + "_" + args.model

if args.dataset is None:
    datasets = sorted(os.listdir(args.data_dir))
    split_seeds = range(1, 6)
else:
    datasets = [args.dataset]
    split_seeds = [args.split_seed]

datasets = utils.split_tasks(datasets, args.cpu, args.group)

for i, dataset in enumerate(datasets):
    print(i, len(datasets), dataset)
    result_dir = utils.makedir([args.result_dir, dataset, args.method])
    tic = time.time()
    run_baseline(args.data_dir, dataset, result_dir, space, params, args.model, args.method)
    utils.save_prep_space(space, result_dir)
