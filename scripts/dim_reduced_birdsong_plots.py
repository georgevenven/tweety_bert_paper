import torch
import os
import sys

sys.path.append("src")

from utils import load_model, detailed_count_parameters
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')
  
from analysis import plot_umap_projection, ComputerClusterPerformance, plot_metrics, sliding_window_umap


# weights_path = "experiments/Yarden_Only_128/saved_weights/model_step_28000.pth"
# config_path = "experiments/Yarden_Only_128/config.json"

# model = load_model(config_path, weights_path)
# model = model.to(device)

# # TweetyBERT 128 Step Generated
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/media/george-vengrovski/disk1/yarden_128step_test",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-1, 
# dict_key="attention_output", 
# context=1000, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="128_Step_Trained_Model_attention-1,500k",
# )

weights_path = "experiments/OG_Yarden_Only_128/saved_weights/model_step_19000.pth"
config_path = "experiments/OG_Yarden_Only_128/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

# TweetyBERT 128 OG Model 
plot_umap_projection(
model=model, 
device=device, 
data_dir="/media/george-vengrovski/disk1/yarden_OG_llb3",
samples=5e4, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=-1, 
dict_key="attention_output", 
context=1000, 
raw_spectogram=False,
save_dict_for_analysis = False,
save_name="LLB3_Yarden_Trained_Model_attention-1,5k_test",
)

# # TweetyBERT 128 OG Model 
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/media/george-vengrovski/disk1/yarden_OG_llb16",
# samples=5e3, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-1, 
# dict_key="attention_output", 
# context=1000, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="LLB16_128_OGStep_Trained_Model_attention-1,5k_TEST",
# )

# # TweetyBERT 128 OG Model 
# plot_umap_projection(
# model=model, 
# device=device, 
# data_dir="/media/george-vengrovski/disk1/yarden_OG_llb11",
# samples=5e5, 
# file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
# layer_index=-1, 
# dict_key="attention_output", 
# context=1000, 
# raw_spectogram=False,
# save_dict_for_analysis = False,
# save_name="LLB11_128_OGStep_Trained_Model_attention-1,500k",
# )


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


# cluster_performance = ComputerClusterPerformance(labels_path = ["/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_128_OGStep_Trained_Model_attention-1,50k.npz","/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_128_Step_Trained_Model_attention-1,500k.npz"])
# metrics = cluster_performance.compute_vmeasure_score()

# print(metrics)

# plot_metrics(metrics, ["OG","generated"])