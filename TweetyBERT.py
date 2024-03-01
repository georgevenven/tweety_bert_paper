import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("src")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from experiment_manager import ExperimentRunner

#Initialize experiment runner
experiment_runner = ExperimentRunner(device="cuda")

# Define configurations
configurations = [
        {"experiment_name": "Budgie_test", "loss_function": "mse_loss", "train_dir": "files/combined_cornell_budgie_train", "test_dir": "files/combined_cornell_budgie_test", "batch_size": 48, "d_transformer": 196, "nhead_transformer": 4, "spec_height": 513, "dropout": 0.2, "dim_feedforward": 768, "transformer_layers": 4, "m": 50, "p": 0.004, "alpha": 1, "pos_enc_type": "relative", "learning_rate": 3e-4, "max_steps": 25e3, "eval_interval": 1000, "save_interval": 1000, "context": 1000, "weight_decay": 0, "tau":1,  "early_stopping": True, "patience": 4, "trailing_avg_window":200, "num_ground_truth_labels": 30},
         {"experiment_name": "Canary_test_new_spec_generation_process", "loss_function": "mse_loss", "train_dir": "files/canary_no_clip_full_freq_train", "test_dir": "files/canary_no_clip_full_freq_test", "batch_size": 48, "d_transformer": 196, "nhead_transformer": 4, "spec_height": 513, "dropout": 0.2, "dim_feedforward": 768, "transformer_layers": 4, "m": 50, "p": 0.004, "alpha": 1, "pos_enc_type": "relative", "learning_rate": 3e-4, "max_steps": 25e3, "eval_interval": 1000, "save_interval": 1000, "context": 1000, "weight_decay": 0, "tau":1,  "early_stopping": True, "patience": 4, "trailing_avg_window":200, "num_ground_truth_labels": 30}
]

for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
