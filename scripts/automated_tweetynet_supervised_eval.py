import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


sys.path.append("src")
os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

from TweetyNET import frame_error_rate, TweetyNET_Dataset, TweetyNetTrainer, CollateFunction, TweetyNet, ModelEvaluator

train_dir = '/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/yarden_train'
test_dir = '/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/yarden_test'

train_dataset = TweetyNET_Dataset(train_dir, num_classes=30)
test_dataset = TweetyNET_Dataset(test_dir, num_classes=30)

collate_fn = CollateFunction(segment_length=370)  # Adjust the segment length if needed

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


model = TweetyNet(num_classes=30, input_shape=(1, 513, 370))
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2, weight_decay=0.0)

# Initialize the TweetyNetTrainer
trainer = TweetyNetTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    optimizer=optimizer,
    desired_total_steps=21,  # Set your desired total steps
    patience=4  # Set your patience for early stopping
)

# Start the training process
trainer.train()
trainer.plot_results()

# Initialize the ModelEvaluator with the test_loader and the trained model
evaluator = ModelEvaluator(test_loader=test_loader, model=model, device=device)

# Validate the model. This method should return the class-wise and total frame error rates
class_frame_error_rates, total_frame_error_rate = evaluator.validate_model_multiple_passes()

# Save the results to a file for later inspection
evaluator.save_results("evaluation_results.json")

# Plot the error rates for visual inspection
evaluator.plot_error_rates("error_rates_plot.png")