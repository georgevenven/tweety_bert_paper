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
        {
        "experiment_name": "Yarden_Spec_3birds",
        "continue_training": False,
        "train_dir": "files/yarden_train",
        "test_dir": "files/yarden_test",
        "batch_size": 48,
        "d_transformer": 196,
        "nhead_transformer": 4,
        "num_freq_bins": 513,
        "dropout": 0.2,
        "dim_feedforward": 768,
        "transformer_layers": 4,
        "m": 50,
        "p": 0.004,
        "alpha": .95,
        "pos_enc_type": "relative",
        "learning_rate": 3e-4,
        "max_steps": 5e5,
        "eval_interval": 50,
        "save_interval": 1000,
        "context": 1000,
        "weight_decay": 0,
        "early_stopping": True,
        "patience": 8,
        "trailing_avg_window": 200,
        "num_ground_truth_labels": 50
        },
        # {
        # "experiment_name": "NoClip_Spec_3birds",
        # "continue_training": False,
        # "train_dir": "files/canary_no_clip_full_freq_train",
        # "test_dir": "files/canary_no_clip_full_freq_test",
        # "batch_size": 48,
        # "d_transformer": 196,   
        # "nhead_transformer": 4,
        # "num_freq_bins": 513,
        # "dropout": 0.2,
        # "dim_feedforward": 768,
        # "transformer_layers": 4,
        # "m": 50,
        # "p": 0.004,
        # "alpha": .95,
        # "pos_enc_type": "relative",
        # "learning_rate": 3e-4,
        # "max_steps": 5e5,
        # "eval_interval": 500,
        # "save_interval": 2500,
        # "context": 1000,
        # "weight_decay": 0,
        # "early_stopping": True,
        # "patience": 8,
        # "trailing_avg_window": 200,
        # "num_ground_truth_labels": 50
        # },
        {
        "experiment_name": "Yarden_Spec_3birds_alpha_.5",
        "continue_training": False,
        "train_dir": "files/yarden_train",
        "test_dir": "files/yarden_test",
        "batch_size": 48,
        "d_transformer": 196,
        "nhead_transformer": 4,
        "num_freq_bins": 513,
        "dropout": 0.2,
        "dim_feedforward": 768,
        "transformer_layers": 4,
        "m": 50,
        "p": 0.004,
        "alpha": .5,
        "pos_enc_type": "relative",
        "learning_rate": 3e-4,
        "max_steps": 5e5,
        "eval_interval": 50,
        "save_interval": 1000,
        "context": 1000,
        "weight_decay": 0,
        "early_stopping": True,
        "patience": 8,
        "trailing_avg_window": 200,
        "num_ground_truth_labels": 50
        },
#         {
#         "experiment_name": "pause_test",
#         "continue_training": True,
#         "max_steps": 100,
#         },
]


for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
