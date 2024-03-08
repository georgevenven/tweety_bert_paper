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

    def generate_train_test(self, file_min_size=1e3, file_limit_size=1e4):
        files = os.listdir(self.data_root)
        for file in tqdm(files, desc="Processing Files"):
            if file.endswith(".npz"):
                try:
                    f = np.load(os.path.join(self.data_root, file), allow_pickle=True)
                    spectrogram = f['s']
                    spectrogram[np.isnan(spectrogram)] = 0

                    if 'labels' in f:
                        labels = f['labels']
                    else:
                        labels = None

                    f_dict = {'s': spectrogram, 'labels': labels}
                    segment_filename = f"{os.path.splitext(file)[0]}{os.path.splitext(file)[1]}"
                    save_path = os.path.join(self.train_dir if np.random.uniform() < self.train_prop else self.test_dir, segment_filename)
                    np.savez(save_path, **f_dict)

                    # num_segments = max(1, int(np.ceil(spectrogram.shape[1] / file_limit_size)))
                    # for segment_index in range(num_segments):
                    #     start = int(segment_index * file_limit_size)  # Ensure start is an integer
                    #     end = int(min((segment_index + 1) * file_limit_size, spectrogram.shape[1]))  # Ensure end is an integer
                        
                    #     if (end - start) < file_min_size:
                    #         continue  # Skip segments smaller than the minimum size

                    #     segment_spectrogram = spectrogram[:, start:end]
                    #     if labels is not None:
                    #         segment_labels = labels[start:end]  # Assuming labels can be indexed in the same way
                    #         f_dict = {'s': segment_spectrogram, 'labels': segment_labels}
                    #     else:
                    #         f_dict = {'s': segment_spectrogram}

                    #     # Generate a unique filename for each segment
                    #     segment_filename = f"{os.path.splitext(file)[0]}_segment{segment_index}{os.path.splitext(file)[1]}"
                    #     save_path = os.path.join(self.train_dir if np.random.uniform() < self.train_prop else self.test_dir, segment_filename)
                        
                    #     np.savez(save_path, **f_dict)
                except Exception as e:
                    print(f"Error processing {file}: {e}")