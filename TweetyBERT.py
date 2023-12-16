import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("src")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

from experiment_manager import ExperimentRunner

#Initialize experiment runner
experiment_runner = ExperimentRunner(device="cuda")

# Define configurations
configurations = [
        {"experiment_name": "TweetyBERT-Cluster-20k-Tau-1-WeightDecay0-wsilence-2000-context", "notes": "Why break apart?", "loss_function": "compute_loss", "plot_umap": False, "train_dir": "train", "test_dir": "test", "batch_size": 16, "d_transformer": 384, "nhead_transformer": 4, "embedding_dim": 50, "num_clusters": 100, "dropout": 0.1, "dim_feedforward": 1536, "transformer_layers": 4, "m": 10, "p": 0.015, "alpha": 1, "sigma": 10, "learning_rate": 3e-4, "max_steps": 5e4, "eval_interval": 1000, "save_interval": 9999, "umap_data_dir": "umap_eval_dataset_llb16", "remove_silences": False, "subsample": 1, "context": 2000, "time_bins_umap_point": 100, "weight_decay": 0, "tau":1}
]

for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
