import matplotlib.pyplot as plt
import os
import sys

# so relative paths can be used 
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

sys.path.append("src")

from spectogram_processor import SpectrogramProcessor

# Define a configuration class or use a dictionary
class Config:
    def __init__(self, data_root, train_dir, test_dir, n_clusters):
        self.data_root = data_root
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.n_clusters = n_clusters
        
configs = [
    Config(data_root="/media/george-vengrovski/disk2/canary_temp/canary_spec", train_dir="files/canary_no_clip_full_freq_train", test_dir="files/canary_no_clip_full_freq_test", n_clusters=30)
]

# Iterate over the configurations and process
for config in configs:
    processor = SpectrogramProcessor(data_root=config.data_root, train_dir=config.train_dir, test_dir=config.test_dir, n_clusters=config.n_clusters)

    processor.clear_directory(config.train_dir)
    processor.clear_directory(config.test_dir)

    processor.generate_train_test()