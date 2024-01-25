import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("src")

from utils import load_model, detailed_count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

# Plotting PCA, UMAP, and TweetyBERT plots of birdsong samples 

weights_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/temp_storage/TweetyBERT-MSE_LLB3_50_Mask/saved_weights/model_step_7400.pth"
config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/temp_storage/TweetyBERT-MSE_LLB3_50_Mask/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

from analysis import plot_umap_projection

plot_umap_projection(
model=model, 
device=device, 
data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test",
remove_silences=True,  # Using new config parameter``
samples=5e5, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=-2, 
dict_key="V", 
time_bins_per_umap_point=1, 
context=1000,  # Using new config parameter
raw_spectogram=False,
save_dict_for_analysis = False,
save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/test.png",
compute_svm= False,
color_scheme = "Labels"
)

# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test",
# remove_silences=False,  # Using new config parameter``
# samples=5e4, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-2, 
# dict_key="V", 
# time_bins_per_umap_point=1, 
# context=1000,  # Using new config parameter
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/mse_2.png"
# )

# weights_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/TweetyBERT-cluster_LLB3_10_Mask_50_clusters/saved_weights/model_step_8600.pth"
# config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/TweetyBERT-cluster_LLB3_10_Mask_50_clusters/config.json"

# model = load_model(config_path, weights_path)
# model = model.to(device)

# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test_50",
# remove_silences=False,  # Using new config parameter``
# samples=5e4, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-1, 
# dict_key="V", 
# time_bins_per_umap_point=1, 
# context=1000,  # Using new config parameter
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/cluster_1.png"
# )

# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test_50",
# remove_silences=False,  # Using new config parameter``
# samples=5e4, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-1, 
# dict_key="V", 
# time_bins_per_umap_point=1, 
# context=1000,  # Using new config parameter
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/cluster_2.png"
# )
