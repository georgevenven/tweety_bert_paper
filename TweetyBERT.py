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

configurations = [
        # {
        # "experiment_name": "pause_test",
        # "continue_training": False,
        # "train_dir": "files/canary_no_clip_full_freq_train",
        # "test_dir": "files/canary_no_clip_full_freq_test",
        # "batch_size": 8,
        # "d_transformer": 196,
        # "nhead_transformer": 4,
        # "spec_height": 513,
        # "dropout": 0.2,
        # "dim_feedforward": 768,
        # "transformer_layers": 4,
        # "m": 10,
        # "p": 0.004,
        # "alpha": .5,
        # "pos_enc_type": "relative",
        # "learning_rate": 3e-4,
        # "max_steps": 500,
        # "eval_interval": 50,
        # "save_interval": 100,
        # "context": 100,
        # "weight_decay": 0,
        # "early_stopping": True,
        # "patience": 14,
        # "trailing_avg_window": 200,
        # "num_ground_truth_labels": 30
        # },
        {
        "experiment_name": "pause_test",
        "continue_training": True,
        "max_steps": 1000,
        },
]


for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
