import os
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


class SpectrogramProcessor:
    def clear_directory(self, directory_path):
        """Deletes all files and subdirectories within a specified directory."""
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
    def __init__(self, data_root, train_dir, test_dir, train_prop=0.8, model=None, device=None):
        self.data_root = data_root
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_prop = train_prop
        self.kmeans = None 

        # Existing initialization code...
        self.model = model
        self.device = device

        # Create directories if they don't exist
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)

        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

    def generate_train_test(self):
        files = os.listdir(self.data_root)
        for i, file in tqdm(enumerate(files), total=len(files), desc="Processing Files"):
            if file.endswith(".npz"):
                try:
                    f = np.load(os.path.join(self.data_root, file), allow_pickle=True)
                    spectogram = f['s']
                    spectogram[np.isnan(spectogram)] = 0

                    if 'labels' in f:
                        f_dict = {'s': spectogram, 'labels': f['labels']}
                    else:
                        f_dict = {'s': spectogram}

                    segment_filename = file

                    # Decide where to save the segmented file
                    if np.random.uniform() < self.train_prop:
                        save_path = os.path.join(self.train_dir, segment_filename)
                    else:
                        save_path = os.path.join(self.test_dir, segment_filename)

                    np.savez(save_path, **f_dict)
                except:
                    continue