import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("src")

from utils import load_model, detailed_count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# device = torch.device("cuda")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

# Plotting PCA, UMAP, and TweetyBERT plots of birdsong samples 

weights_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/TweetyBERT-MSE-Mask-Before-50-mask-alpha-1/saved_weights/model_step_6400.pth"
config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/TweetyBERT-MSE-Mask-Before-50-mask-alpha-1/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)
  
from analysis import plot_umap_projection, similarity_of_vectors

plot_umap_projection(
model=model, 
device=device, 
data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test",
remove_silences=False,  # Using new config parameter``
samples=5e5, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=1, 
dict_key="attention_output", 
time_bins_per_umap_point=1, 
context=1000,  # Using new config parameter98
raw_spectogram=True,
save_dict_for_analysis = True,
save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/raw_spec_phrase_labels.png",
compute_svm= False,
color_scheme = "Label"
)

# similarity_of_vectors(model, device, data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test",
#                          remove_silences=False, samples=10, file_path='category_colors_llb3.pkl.pkl', 
#                          layer_index=1, dict_key="V",
#                          context=1000, save_dir=None, raw_spectogram=False)


