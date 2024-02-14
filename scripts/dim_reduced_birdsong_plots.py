import torch
import os
import sys

sys.path.append("src")

from utils import load_model, detailed_count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

weights_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/TweetyBERT-MSE-Mask-Before-50-mask-alpha-1/saved_weights/model_step_6400.pth"
config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/TweetyBERT-MSE-Mask-Before-50-mask-alpha-1/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)
  
from analysis import plot_umap_projection

plot_umap_projection(
model=model, 
device=device, 
data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test",
remove_silences=False,  # Using new config parameter``
samples=5e6, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=1, 
dict_key="attention_output", 
time_bins_per_umap_point=1, 
context=1000,  # Using new config parameter98
raw_spectogram=False,
save_dict_for_analysis = True,
save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/hdbscanformatlab.png",
compute_svm= False,
color_scheme = "Label"
)

