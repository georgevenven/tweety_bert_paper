import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json 

sys.path.append("src")

from utils import load_model, detailed_count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

# Loading the model  
tweetyBERT_config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/200k_steps_all_canaries_half_hubert/config.json"
tweetyBERT_weight_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/200k_steps_all_canaries_half_hubert/saved_weights/model_step_399960.pth"

with open(tweetyBERT_config_path, 'r') as f:
    config = json.load(f)  # Load and parse the JSON file

model = load_model(tweetyBERT_config_path, tweetyBERT_weight_path, device)
eval_dataset_path = "/home/george-vengrovski/Documents/data/eval_dataset/llb3_data_matrices"

# load eval set into dataclass 

from torch.utils.data import DataLoader
from data_class SongDataSet_Image, CollateFunction

train_dir = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/train"
test_dir = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/test"

train_dataset = SongDataSet_Image(train_dir)
test_dataset = SongDataSet_Image(test_dir)

collate_fn = CollateFunction(segment_length=1000)  # Adjust the segment length if needed

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# train the model 

# evaluate the results