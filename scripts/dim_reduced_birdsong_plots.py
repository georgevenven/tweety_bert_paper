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

weights_path = "experiments/Yarden_FreqTruncated/saved_weights/model_step_17000.pth"
config_path = "experiments/Yarden_FreqTruncated/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

# TweetyBERT 128 OG Model 
plot_umap_projection(
model=model, 
device=device, 
data_dir="/media/george-vengrovski/disk1/yarden_OG_llb16",
samples=5e5, 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index=-2, 
dict_key="attention_output", 
context=1000, 
raw_spectogram=False,
save_dict_for_analysis = False,
save_name="LLB16_Yarden_FreqTruncated_Trained_Model_attention-2_500k",
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



# cluster_performance = ComputerClusterPerformance(labels_path = ["/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_128_OGStep_Trained_Model_attention-1,50k.npz","/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_128_Step_Trained_Model_attention-1,500k.npz"])
# metrics = cluster_performance.compute_vmeasure_score()

# print(metrics)

# plot_metrics(metrics, ["OG","generated"])