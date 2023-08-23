import numpy as np
import pandas as pd
import torch
import os
import json
import shutil
import pickle
from matplotlib import pyplot as plt
from torch.distributions.utils import logits_to_probs
from collections import defaultdict

def makedir(dir_list, file=None, remove_old_dir=False):
    save_dir = os.path.join(*dir_list)

    if remove_old_dir and os.path.exists(save_dir) and file is None:
        shutil.rmtree(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir

def save_prep_space(prep_space, save_dir):
    prep_space_dict = {}

    for step in prep_space[1:]:
        tf_options = [tf.method for tf in step["tf_options"]]
        prep_space_dict[step["name"]] = tf_options

    with open(os.path.join(save_dir, "prep_space.json"), "w") as f:
        json.dump(prep_space_dict, f, indent=4)


def print_params_grad(model):
    for k, v in model.named_parameters():
        print(k)
        print(v.grad)

class SummaryWriter(object):
    def __init__(self):
        self.scalar_logging = {}
        self.tf_probs_logging = {}
        self.pipeline_logging = []
        self.alpha_logging = []

    def logging(self, name, x, global_step, logging_dict):
        if name not in logging_dict:
            logging_dict[name] = {"indices":[global_step], "values":[x]}
        else:
            logging_dict[name]["indices"].append(global_step)
            logging_dict[name]["values"].append(x)

    def add_scalar(self, name, x, global_step):
        self.logging(name, x, global_step, self.scalar_logging)

    def add_tf_probs(self, name, x, global_step):
        self.logging(name, x, global_step, self.tf_probs_logging)

    def plot_scalars(self, log_dir):
        save_dir = makedir([log_dir, "scalar_figures"])
        for name, scalar in self.scalar_logging.items():
            plt.plot(scalar["indices"], scalar["values"])
            plt.xlabel("epoch")
            plt.ylabel(name)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "{}.png".format(name)))
            plt.clf()

    def save_logging(self, log_dir):
        with open(os.path.join(log_dir, "scalar_logging.json"), "w") as f:
            json.dump(self.scalar_logging, f, indent=4)

        with open(os.path.join(log_dir, "tf_probs_logging.p"), "wb") as f:
            pickle.dump(self.tf_probs_logging, f)

        pipeline_logging = pd.DataFrame(self.pipeline_logging)
        pipeline_logging.to_csv(os.path.join(log_dir, "pipeline_logging.csv"), index=False)
        
        if len(self.alpha_logging) > 0:
            alpha_logging = pd.DataFrame(np.array(self.alpha_logging))
            alpha_logging.to_csv(os.path.join(log_dir, "alpha_logging.csv"), index=False)

    def save(self, log_dir):
        self.save_logging(log_dir)
        self.plot_scalars(log_dir)

    def add_pipeline(self, pipeline, global_step):
        first_transformer = pipeline[0]
        num_tf_probs = None
        cat_tf_probs = None
        log = defaultdict(list)

        log["epoch"] = global_step

        if first_transformer.num_tf_prob_logits is not None:
            num_tf_probs = logits_to_probs(first_transformer.num_tf_prob_logits.detach().data).numpy()
            self.add_tf_probs('MVImputer num tf probs', num_tf_probs, global_step=global_step)
            num_tf_methods = first_transformer.num_tf_methods
            best_num_tf_idx = np.argmax(num_tf_probs, axis=1)
            for i, idx in enumerate(best_num_tf_idx):
                log[first_transformer.feature_names[i]].append("{}:{:.2f}".format(num_tf_methods[idx], num_tf_probs[i, idx]))

        if first_transformer.cat_tf_prob_logits is not None:
            cat_tf_probs = logits_to_probs(first_transformer.cat_tf_prob_logits.detach().data).numpy()
            self.add_tf_probs('MVImputer cat tf probs', cat_tf_probs, global_step=global_step)
            cat_tf_methods = first_transformer.cat_tf_methods
            best_cat_tf_idx = np.argmax(cat_tf_probs, axis=1)
            for i, idx in enumerate(best_cat_tf_idx):
                feature_idx = i+first_transformer.out_num_features
                log[first_transformer.feature_names[feature_idx]].append("{}:{:.2f}".format(cat_tf_methods[idx], cat_tf_probs[i, idx]))

        for transformer in pipeline[1:]:
            transformer_name = transformer.name
            tf_probs = logits_to_probs(transformer.tf_prob_logits.detach().data).numpy()
            self.add_tf_probs('Transformer {} tf probs'.format(transformer_name), tf_probs, global_step=global_step)
            tf_methods = transformer.tf_methods
            best_tf_idx = np.argmax(tf_probs, axis=1)

            for i, idx in enumerate(best_tf_idx):
                log[first_transformer.feature_names[i]].append("{}:{:.2f}".format(tf_methods[idx], tf_probs[i, idx]))

        self.pipeline_logging.append(log)

    def add_pipeline_alpha(self, alpha, global_step):
        self.alpha_logging.append(alpha.detach().cpu().numpy().reshape(-1))

    def add_baseline_pipeline(self, pipeline, global_step):
        first_transformer = pipeline[0]
        num_tf_probs = None
        cat_tf_probs = None
        log = defaultdict(list)

        log["epoch"] = global_step

        if first_transformer.num_tf_probs is not None:
            num_tf_probs = first_transformer.num_tf_probs
            self.add_tf_probs('MVImputer num tf probs', first_transformer.num_tf_probs, global_step=global_step)
            num_tf_methods = [tf.method for tf in first_transformer.num_tf_options]
            best_num_tf_idx = np.argmax(num_tf_probs, axis=1)
            for i, idx in enumerate(best_num_tf_idx):
                log[first_transformer.feature_names[i]].append("{}:{:.2f}".format(num_tf_methods[idx], num_tf_probs[i, idx]))

        if first_transformer.cat_tf_probs is not None:
            cat_tf_probs = first_transformer.cat_tf_probs
            self.add_tf_probs('MVImputer cat tf probs', cat_tf_probs, global_step=global_step)
            cat_tf_methods = [tf.method for tf in first_transformer.cat_tf_options]
            best_cat_tf_idx = np.argmax(cat_tf_probs, axis=1)
            for i, idx in enumerate(best_cat_tf_idx):
                feature_idx = i+first_transformer.out_num_features
                log[first_transformer.feature_names[feature_idx]].append("{}:{:.2f}".format(cat_tf_methods[idx], cat_tf_probs[i, idx]))

        for transformer in pipeline[1:]:
            transformer_name = transformer.name
            tf_probs = transformer.tf_probs
            self.add_tf_probs('Transformer {} tf probs'.format(transformer_name), tf_probs, global_step=global_step)
            tf_methods = [tf.method for tf in transformer.tf_options]
            best_tf_idx = np.argmax(tf_probs, axis=1)

            for i, idx in enumerate(best_tf_idx):
                log[first_transformer.feature_names[i]].append("{}:{:.2f}".format(tf_methods[idx], tf_probs[i, idx]))

        self.pipeline_logging.append(log)

def split_tasks(tasks, num_cpu, cpu_id):
    group_indices = np.array_split(np.arange(len(tasks)), num_cpu)
    indices = group_indices[cpu_id]
    subtasks = [tasks[i] for i in indices]
    return subtasks