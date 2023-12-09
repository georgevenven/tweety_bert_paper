import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import os
import torch
from torch.utils.data import DataLoader
import hashlib
import json
import shutil
import sys

sys.path.append("src")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_cluster/project')

from data_class import SongDataSet_Image, CollateFunction
from model import TweetyBERT
from analysis import plot_umap_projection
from utils import detailed_count_parameters


# programatically load last saved weight
weights = '/home/george-vengrovski/Documents/projects/tweety_bert_cluster/project/experiments/Attempting to train on all three birds to completion (50k steps)/saved_weights/model_step_40000.pth'
config = '/home/george-vengrovski/Documents/projects/tweety_bert_cluster/project/experiments/Attempting to train on all three birds to completion (50k steps)/config.json'



# Load the configuration file
config_path = '/home/george-vengrovski/Documents/projects/tweety_bert_cluster/project/experiments/Attempting to train on all three birds to completion (50k steps)/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)  # Load and parse the JSON file

 # Data Loading
collate_fn = CollateFunction(segment_length=config['context'])
train_dataset = SongDataSet_Image(config['train_dir'], num_classes=config['num_clusters'], subsampling=True, subsample_factor=config['subsample'], remove_silences=config['remove_silences'])
test_dataset = SongDataSet_Image(config['test_dir'], num_classes=config['num_clusters'], subsampling=True, subsample_factor=config['subsample'], remove_silences=config['remove_silences'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=16)

# Initialize model
model = TweetyBERT(
    d_transformer=config['d_transformer'], 
    nhead_transformer=config['nhead_transformer'],
    embedding_dim=config['embedding_dim'],
    num_labels=config['num_clusters'],
    tau=None,
    dropout=config['dropout'],
    dim_feedforward=config['dim_feedforward'],
    transformer_layers=config['transformer_layers'],
    m=config['m'],
    p=config['p'],
    alpha=config['alpha'],
    sigma=config['sigma']
).to(device)

detailed_count_parameters(model)


##### 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# Assuming train_loader is your DataLoader
data_iter = iter(train_loader)
spec, _, _ = next(data_iter)

# Move the sample to the appropriate device
spec= spec.to(device)

# Pass the sample through the model
_, layer_outputs = model.inference_forward(spec)

# Iterate through each layer's output and plot PCA
for i, output in enumerate(layer_outputs):
    # Assuming each output is a dictionary with 'feed_forward_output' key
    layer_data = output['feed_forward_output']

    print(layer_data.shape)

    # Reshape or aggregate the data for PCA
    # This depends on the shape of your data; you might need to adjust this part
    pca_input = layer_data.view(layer_data.size(0), -1).cpu().detach().numpy()

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_input)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title(f'PCA of Layer {i+1}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()