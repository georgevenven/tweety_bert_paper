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
        {"experiment_name": "MSE_Test", "loss_function": "mse_loss", "train_dir": "files/llb3_train_25", "test_dir": "files/llb3_test_25", "batch_size": 48, "d_transformer": 196, "nhead_transformer": 4, "embedding_dim": 196, "num_clusters": 25, "dropout": 0.1, "dim_feedforward": 768, "transformer_layers": 4, "m": 100, "p": 0.001, "alpha": .5, "pos_enc_type": "relative", "learning_rate": 1e-4, "max_steps": 1e3, "eval_interval": 100, "save_interval": 9999, "remove_silences": False, "context": 1000, "weight_decay": 0, "tau":1,  "early_stopping": True, "patience": 4, "trailing_avg_window":1000}
]
for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
