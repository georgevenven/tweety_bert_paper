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
        {"experiment_name": "Carrot-1", "loss_function": "mse_loss", "train_dir": "files/warble_train", "test_dir": "files/warble_test", "batch_size": 48, "d_transformer": 196, "nhead_transformer": 4, "embedding_dim": 196, "num_clusters": 50, "dropout": 0.2, "dim_feedforward": 768, "transformer_layers": 4, "m": 50, "p": 0.004, "alpha": 1, "pos_enc_type": "relative", "learning_rate": 3e-4, "max_steps": 25e3, "eval_interval": 500, "save_interval": 1000, "remove_silences": False, "context": 1000, "weight_decay": 0, "tau":1,  "early_stopping": True, "patience": 4, "trailing_avg_window":200}
      
]

for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
