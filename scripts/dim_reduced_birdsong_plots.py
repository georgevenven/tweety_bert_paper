import torch
import os
import sys

sys.path.append("src")

from utils import load_model, detailed_count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

weights_path = "experiments/Budgie_Test/saved_weights/model_step_2500.pth"
config_path = "experiments/Budgie_Test/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)
  
from analysis import plot_umap_projection, ComputerClusterPerformance, plot_metrics, sliding_window_umap

# TweetyBERT
plot_umap_projection(
model=model,    
device=device, 
data_dir="files/budgie_test",
samples=5e4, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=2, 
dict_key="V", 
context=1000, 
raw_spectogram=False,
save_dict_for_analysis = True,
save_name="Budgie_Test",
)

# # TweetyBERT
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="files/yarden_llb3_test",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=2, 
# dict_key="V", 
# context=1000, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="yarden_normal_size_V_2_5e5",
# )

# # TweetyBERT
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="files/yarden_llb3_test",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=2, 
# dict_key="attention_output", 
# context=1000, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="yarden_normal_size_attention_2_5e5",
# )

weights_path = "experiments/Budgie_Test_10_Mask/saved_weights/model_step_25000.pth"
config_path = "experiments/Budgie_Test_10_Mask/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

# TweetyBERT
plot_umap_projection(
model=model, 
device=device, 
data_dir="files/noclip_test",
samples=5e4, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=2, 
dict_key="V", 
context=1000, 
raw_spectogram=False,
save_dict_for_analysis = False,
save_name="budgie_test",
)

# # TweetyBERT
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="files/yarden_llb3_test",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=2, 
# dict_key="attention_output", 
# context=1000, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="yarden_double_size_attention_2_5e5",
# )

# # Raw Spectogram 
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="files/llb3_test",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=1, 
# dict_key="attention_output", 
# context=1000, 
# raw_spectogram=True,
# save_dict_for_analysis = False,
# save_name="UMAP_LLB3",
# )

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
# data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/llb3_test",
# remove_silences=False,  # Using new config parameter``
# samples=5e4, ## Excessive to collect all the songs in test set 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=1, 
# dict_key="attention_output", 
# context=1000,  # Using new config parameter98
# raw_spectogram=True,
# save_dict_for_analysis = True,
# save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/RAW_UMAP_Windowed-10.png",
# compute_svm= False,
# color_scheme = "Label",
# window_size=10,
# stride=9
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


