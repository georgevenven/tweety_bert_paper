import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json 
import re 
sys.path.append("src")
os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

from utils import load_model, detailed_count_parameters
from linear_probe import LinearProbeModel

def load_config_from_path(path):
    with open(path, 'r') as f:
        config = json.load(f)  # Load and parse the JSON file
    return config 

# eval_dataset_path = "/home/george-vengrovski/Documents/data/eval_dataset/llb3_data_matrices"

# # load eval set into dataclass 

# from torch.utils.data import DataLoader
# from data_class SongDataSet_Image, CollateFunction

# train_dir = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/train"
# test_dir = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/test"

# train_dataset = SongDataSet_Image(train_dir)
# test_dataset = SongDataSet_Image(test_dir)

# collate_fn = CollateFunction(segment_length=1000)  # Adjust the segment length if needed

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# # The Flow
# # Select Models that You Will Be Analyzing (experiments folder)

# evaluate the FER on all the layers
def probe_eval(model, results_path, folder):
    # Step 1: Create directory in results_path named folder
    folder_path = os.path.join(results_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # create psuedo input to extract the num layers / each 
        
    print(model.get_layer_output_pairs())

    pass 

def get_highest_numbered_file(files):
    weights_files = [f for f in files if re.match(r'model_step_\d+\.pth', f)]
    if not weights_files:
        return None
    highest_num = max([int(re.search(r'\d+', f).group()) for f in weights_files])
    return f'model_step_{highest_num}.pth'

def execute_eval_of_experiments(base_path, results_path):
    # loop through all experimental folders 
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder != 'archive':
            weights_folder = os.path.join(folder_path, 'saved_weights')
            config_path = os.path.join(folder_path, 'config.json')

            if os.path.exists(weights_folder):
                files = os.listdir(weights_folder)
                weights_file = get_highest_numbered_file(files)
                
                if weights_file and config_path:
                    weight_path = os.path.join(weights_folder, weights_file)
                    model = load_model(config_path, weight_path, device)
                    probe_eval(model, results_path, folder)  
                    break
                else: 
                    print("files for eval of experiment not found")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}") 

experiment_paths = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments"
results_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/results"
execute_eval_of_experiments(experiment_paths, results_path)
