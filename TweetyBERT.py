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
        "experiment_name": "Yarden_Only",
        "continue_training": False,
        "train_dir": "/media/george-vengrovski/disk1/yarden_train",
        "test_dir": "/media/george-vengrovski/disk1/yarden_test",
        "batch_size": 36,
        "d_transformer": 196,   
        "nhead_transformer": 4,
        "num_freq_bins": 513,
        "dropout": 0.2,
        "dim_feedforward": 768,
        "transformer_layers": 3,
        "m": 100,
        "p": 0.01,
        "alpha": 1,
        "pos_enc_type": "relative",
        "learning_rate": 3e-4,
        "max_steps": 5e6,
        "eval_interval": 500,
        "save_interval": 1000,
        "context": 1000,
        "weight_decay": 0.0,
        "early_stopping": True,
        "patience": 8,
        "trailing_avg_window": 200,
        "num_ground_truth_labels": 50
        }
        # {
        # "experiment_name": "1000_budgie",
        # "continue_training": False,
        # "train_dir": "/media/george-vengrovski/disk1/multispecies_data_set_train",
        # "test_dir": "/media/george-vengrovski/disk1/multispecies_data_set_test",
        # "batch_size": 52,
        # "d_transformer": 384,   
        # "nhead_transformer": 8,
        # "num_freq_bins": 513,
        # "dropout": 0.2,
        # "dim_feedforward": 1536,
        # "transformer_layers": 6,
        # "m": 100,
        # "p": 0.01,
        # "alpha": 1,
        # "pos_enc_type": "relative",
        # "learning_rate": 1e-4,
        # "max_steps": 5e6,
        # "eval_interval": 500,
        # "save_interval": 1000,
        # "context": 500,
        # "weight_decay": 0.0,
        # "early_stopping": True,
        # "patience": 8,
        # "trailing_avg_window": 200,
        # "num_ground_truth_labels": 50
        # }
        # {
        # "experiment_name": "Goliath-0-No_weight_decay_a1_fp16_CVM",
        # "continue_training": True,
        # "train_dir": "/media/george-vengrovski/disk1/multispecies_data_set_train",
        # "test_dir": "/media/george-vengrovski/disk1/budgie_test_specs",
        # "max_steps": 5e10,
        # "eval_interval": 1,
        # "patience": 800,
        # "m": 50
        # },
]


for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
