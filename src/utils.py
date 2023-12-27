
import os
import torch 
from torch.utils.data import DataLoader
import json
import sys
from model import TweetyBERT


def load_weights(dir, model):
    model.load_state_dict(torch.load(dir))

def detailed_count_parameters(model):
    """Print details of layers with the number of trainable parameters in the model."""
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
        # print(f"Layer: {name} | Parameters: {param:,} | Shape: {list(parameter.shape)}")
    print(f"Total Trainable Parameters: {total_params:,}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)  # Load and parse the JSON file

    return config

def load_model(config_path, weight_path):
    config = load_config(config_path)

    # Initialize model
    model = TweetyBERT(
        d_transformer=config['d_transformer'], 
        nhead_transformer=config['nhead_transformer'],
        embedding_dim=config['embedding_dim'],
        num_labels=config['num_clusters'],
        tau=config['tau'],
        dropout=config['dropout'],
        dim_feedforward=config['dim_feedforward'],
        transformer_layers=config['transformer_layers'],
        m=config['m'],
        p=config['p'],
        alpha=config['alpha'],
        pos_enc_type=config['pos_enc_type'],
        length=config['context']
    )

    if weight_path:
        load_weights(dir=weight_path, model=model)

    return model 