import torch
import os
import sys

sys.path.append("src")

from utils import load_model, detailed_count_parameters
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')
  
from analysis import plot_umap_projection, ComputerClusterPerformance, plot_metrics, sliding_window_umap


weights_path = "experiments/Yarden_Only/saved_weights/model_step_500.pth"
config_path = "experiments/Yarden_Only/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

# TweetyBERT
plot_umap_projection(
model=model, 
device=device, 
data_dir="/media/george-vengrovski/disk1/yarden_test",
samples=5e4, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=1, 
dict_key="attention_output", 
context=500, 
raw_spectogram=False,
save_dict_for_analysis = False,
save_name="llb3_labels_attached",
)

# # TweetyBERT
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/media/george-vengrovski/disk1/bf_test_specs",
# samples=5e4, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-3, 
# dict_key="attention_output", 
# context=500, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="bengalese_attn-3",
# )

# # TweetyBERT
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/media/george-vengrovski/disk1/budgie_test_specs",
# samples=5e4, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-3, 
# dict_key="attention_output", 
# context=500, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="budgie_attn-3",
# )

# # TweetyBERT
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/media/george-vengrovski/disk1/brown_thrasher_test_specs",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-3, 
# dict_key="attention_output", 
# context=500, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="brown_thrasher_attn-3",
# )

# # TweetyBERT
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/media/george-vengrovski/disk1/zf_test_specs",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-3, 
# dict_key="attention_output", 
# context=500, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="zf_attn-3",
# )

