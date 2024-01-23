import matplotlib.pyplot as plt
import os
import sys

# so relative paths can be used 
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

sys.path.append("src")

from psuedo_label_generator import SpectrogramProcessor

# Define a configuration class or use a dictionary
class Config:
    def __init__(self, data_root, train_dir, test_dir, n_clusters):
        self.data_root = data_root
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.n_clusters = n_clusters

# # Load configurations from a file or define them in a list
# configs = [
#     Config("/home/george-vengrovski/Documents/data/llb3_data_matrices", "files/llb3_train_50", "files/llb3_test_50", 50),
#     Config("/home/george-vengrovski/Documents/data/llb3_data_matrices", "files/llb3_train_500", "files/llb3_test_500", 500),
#     Config("/home/george-vengrovski/Documents/data/llb11_data_matrices", "files/llb11_train_50", "files/llb11_test_50", 50),
#     Config("/home/george-vengrovski/Documents/data/llb11_data_matrices", "files/llb11_train_500", "files/llb11_test_500", 500),
#     Config("/home/george-vengrovski/Documents/data/llb16_data_matrices", "files/llb16_train_50", "files/llb16_test_50", 50),
#     Config("/home/george-vengrovski/Documents/data/llb16_data_matrices", "files/llb16_train_500", "files/llb16_test_500", 500)
# ]
        
# Load configurations from a file or define them in a list
# configs = [
#     # Config(data_root="/home/george-vengrovski/Documents/data/llb3_data_matrices", train_dir="files/llb3_train", test_dir="files/llb3_test", n_clusters=50),
#     # Config(data_root="/home/george-vengrovski/Documents/data/llb11_data_matrices", train_dir="files/llb11_train", test_dir="files/llb11_test", n_clusters=50),
#     # Config(data_root="/home/george-vengrovski/Documents/data/llb3_data_matrices", train_dir="files/llb3_train_50", test_dir="files/llb3_test_50", n_clusters=50)
# ]
        
configs = [
    # Config(data_root="/home/george-vengrovski/Documents/data/llb3_data_matrices", train_dir="files/llb3_train", test_dir="files/llb3_test", n_clusters=50),
    # Config(data_root="/home/george-vengrovski/Documents/data/llb11_data_matrices", train_dir="files/llb11_train", test_dir="files/llb11_test", n_clusters=50),
    # Config(data_root="/home/george-vengrovski/Documents/data/llb3_data_matrices", train_dir="files/llb3_train_50", test_dir="files/llb3_test_50", n_clusters=50)
]

# Iterate over the configurations and process
for config in configs:
    processor = SpectrogramProcessor(data_root=config.data_root, train_dir=config.train_dir, test_dir=config.test_dir, n_clusters=config.n_clusters)

    processor.clear_directory(config.train_dir)
    processor.clear_directory(config.test_dir)

    processor.generate_train_test()
    processor.generate_embedding(samples=6000)
    closest_features_path = f"files/{config.data_root.split('/')[-1]}_centroid_vectors_{str(config.n_clusters)}.npy"
    processor.find_closest_features_to_centroids(save_path=closest_features_path)
    processor.generate_train_test_labels()