import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("src")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

from psuedo_label_generator import SpectrogramProcessor

data_root = "/home/george-vengrovski/Documents/data/pretrain_dataset"
train = "train"
test = "test"

processor = SpectrogramProcessor(data_root=data_root, train_dir=train, test_dir=test, n_clusters=1000, train_prop=0.8)

# ## CAREFUL
processor.clear_directory(train)
processor.clear_directory(test)
# ## CAREFUL 

processor.generate_train_test()
processor.generate_embedding(samples=5000)
processor.find_closest_features_to_centroids(save_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/closest_spec_vectors.npy")
processor.generate_train_test_labels()