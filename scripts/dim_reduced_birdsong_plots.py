import torch
import os
import sys

sys.path.append("src")

from utils import load_model, detailed_count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

weights_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/Canary-No-Clip-No-Freq/saved_weights/model_step_1000.pth"
config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/Canary-No-Clip-No-Freq/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)
  
from analysis import plot_umap_projection, ComputerClusterPerformance, plot_metrics, sliding_window_umap

plot_umap_projection(
model=model, 
device=device, 
data_dir="files/canary_no_clip_full_freq_test",
remove_silences=False,  # Using new config parameter``
samples=5e4, ## Excessive to collect all the songs in test set 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=1, 
dict_key="attention_output", 
time_bins_per_umap_point=1, 
context=1000,  # Using new config parameter98
raw_spectogram=False,
save_dict_for_analysis = True,
save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/tweety_bert.png",
compute_svm= False,
color_scheme = "Label"
)

# clustering_instance = ComputerClusterPerformance(['/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/500k_run.npz','/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/500k_run.npz', '/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/500k_run.npz'])
# metrics_TweetyBERT = clustering_instance.compute_vmeasure_score()

# clustering_instance = ComputerClusterPerformance(['/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/raw_umap_500k.npz','/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/raw_umap_500k.npz', '/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/raw_umap_500k.npz'])
# metrics_TweetyBERT2 = clustering_instance.compute_vmeasure_score()

# model_list = [metrics_TweetyBERT, metrics_TweetyBERT2]
# model_names = ['TweetyBERT', 'UMAP']

# plot_metrics(model_list, model_names)

# sliding_window_umap(
# model=model, 
# device=device, 
# data_dir="files/warble_test",
# remove_silences=False,  # Using new config parameter``
# samples=5e4, ## Excessive to collect all the songs in test set 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=1, 
# dict_key="attention_output", 
# time_bins_per_umap_point=1, 
# context=1000,  # Using new config parameter98
# raw_spectogram=True,
# save_dict_for_analysis = True,
# save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/UMAP-windowed-test-250.png",
# compute_svm= False,
# color_scheme = "Label",
# window_size=250
# )

# sliding_window_umap(
# model=model, 
# device=device, 
# data_dir="files/warble_test",
# remove_silences=False,  # Using new config parameter``
# samples=5e4, ## Excessive to collect all the songs in test set 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=1, 
# dict_key="attention_output", 
# time_bins_per_umap_point=1, 
# context=1000,  # Using new config parameter98
# raw_spectogram=False,
# save_dict_for_analysis = True,
# save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/TweetyBERT-windowed-test-250.png",
# compute_svm= False,
# color_scheme = "Label",
# window_size=250
# )


