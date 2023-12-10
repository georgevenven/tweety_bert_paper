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
        {"experiment_name": "200k_steps_all_canaries_half_hubert", "notes": "", "loss_function": "compute_loss", "plot_umap": False, "train_dir": "train", "test_dir": "test", "batch_size": 16, "d_transformer": 384, "nhead_transformer": 4, "embedding_dim": 256, "num_clusters": 1000, "dropout": 0.1, "dim_feedforward": 1536, "transformer_layers": 6, "m": 10, "p": 0.015, "alpha": 1, "sigma": 10, "learning_rate": 1e-4, "max_steps": 4e5, "eval_interval": 1000, "save_interval": 9999, "umap_data_dir": "umap_eval_dataset_llb16", "remove_silences": False, "subsample": 1, "context": 1000, "time_bins_umap_point": 100, "weight_decay": 0.0001, "tau":.1}
]

for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
