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

# Plotting PCA, UMAP, and TweetyBERT plots of birdsong samples 
tweetyBERT_config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/0mask/config.json"
tweetyBERT_weight_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/0mask/saved_weights/model_step_49995.pth"
with open(tweetyBERT_config_path, 'r') as f:
    config = json.load(f)  # Load and parse the JSON files

eval_dataset_path = "/home/george-vengrovski/Documents/data/eval_dataset/llb3_data_matrices"

model = load_model(tweetyBERT_config_path, tweetyBERT_weight_path, device)

# single-song analysis (create eval folder with one intresting spectogram via spectogram viewer)

#

# multi-song analysis 
from analysis import plot_umap_projection, plot_pca_projection

plot_umap_projection(
model=model, 
device=device, 
data_dir="/home/george-vengrovski/Documents/data/llb3_data_matrices", 
subsample_factor=config['subsample'],  # Using new config parameter
remove_silences=False,  # Using new config parameter
samples=100, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=-1, 
dict_key="V", 
time_bins_per_umap_point=1, 
context=1000,  # Using new config parameter
raw_spectogram=False,
save_dict_for_analysis = True 
)

# plot_pca_projection(
# model=model, 
# device=device, 
# data_dir="/home/george-vengrovski/Documents/data/eval_dataset/llb3_data_matrices", 
# subsample_factor=config['subsample'],  # Using new config parameter
# remove_silences=False,  # Using new config parameter
# samples=100, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-1, 
# dict_key="intermediate_residual_stream", 
# time_bins_per_pca_point=100, 
# context=config['context'],  # Using new config parameter
# raw_spectogram=False
# )
