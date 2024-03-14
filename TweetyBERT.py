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
        "experiment_name": "Budgie_Test_10_Mask",
        "continue_training": False,
        "train_dir": "files/budgie_train",
        "test_dir": "files/budgie_test",
        "batch_size": 16,
        "d_transformer": 196,   
        "nhead_transformer": 4,
        "num_freq_bins": 257,
        "dropout": 0.2,
        "dim_feedforward": 768,
        "transformer_layers": 4,
        "m": 10,
        "p": 0.02,
        "alpha": 1,
        "pos_enc_type": "relative",
        "learning_rate": 1e-4,
        "max_steps": 5e5,
        "eval_interval": 500,
        "save_interval": 2500,
        "context": 1000,
        "weight_decay": 0,
        "early_stopping": True,
        "patience": 8,
        "trailing_avg_window": 200,
        "num_ground_truth_labels": 50
        },
        # {
        # "experiment_name": "Noclip_Spec_3birds_alpha1_1e-3_normal_size",
        # "continue_training": True,
        # "max_steps": 5e5,
        # "eval_interval": 10,
        # },
]


for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
