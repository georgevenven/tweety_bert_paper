import torch
import os
import numpy as np
import shutil
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import random
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
                
    def __init__(self, data_root, train_dir, test_dir, n_clusters, train_prop=0.8):
        self.data_root = data_root
        self.n_clusters = n_clusters
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_prop = train_prop
        self.kmeans = None 

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
                    labels = f['labels']
                    new_labels = f['labels']

                    # Z-score normalization
                    mean_val = spectogram.mean()
                    std_val = spectogram.std()
                    spectogram = (spectogram - mean_val) / (std_val + 1e-7)  # Adding a small constant to prevent division by zero
                    
                    # Replace NaN values with zeros
                    spectogram[np.isnan(spectogram)] = 0

                    # Crop the spectrogram (assuming this is intended)
                    f_dict = {'s': spectogram[20:216,:], 'labels': labels, 'new_labels': np.zeros(new_labels.shape)}

                    segment_filename = file

                    # Decide where to save the segmented file
                    if np.random.uniform() < self.train_prop:
                        save_path = os.path.join(self.train_dir, segment_filename)
                    else:
                        save_path = os.path.join(self.test_dir, segment_filename)

                    np.savez(save_path, **f_dict)
                except:
                    continue

    def generate_embedding(self, samples=100, logits=False):
        files = [file for file in os.listdir(self.train_dir) if file.endswith(".npz")]
        random.shuffle(files)  # Shuffle the files to ensure random order
        
        # Initialize MiniBatchKMeans
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters)

        processed_samples = 0
        for file in tqdm(files, desc="Fitting K-means", total=min(samples, len(files))):
            if processed_samples >= samples:
                break  # Stop if we have processed the desired number of samples
            
            f = np.load(os.path.join(self.train_dir, file), allow_pickle=True)

            if logits:
                spectogram = f['logits']
            else: 
                spectogram = f['s']
            
            # Reshape the array to a 2D array where each column is a data point for k-means
            spectogram = spectogram.reshape(-1, spectogram.shape[-1]).T  # Transpose to have correct shape for partial_fit
            # Update k-means with the new data
            self.kmeans.partial_fit(spectogram)

            processed_samples += 1  # Increment the count of processed samples
        
        print("K-means training completed")


    def find_closest_features_to_centroids(self, save_path):
        """
        Finds the closest spectrogram slice/features for each k-means centroid and saves them.

        This method iterates through all spectrogram slices in the training data,
        calculates the distance to each k-means centroid, and finds the slice closest to each centroid.
        The closest slices are then saved in a numpy array, where each row represents a centroid.

        Parameters:
        save_path (str): The path to save the numpy file containing the closest features.

        Returns:
        None
        """

        # Check if k-means model exists
        if self.kmeans is None:
            raise ValueError("K-means model has not been initialized.")

        # Initialize a variable to store the closest features for each centroid
        closest_features = np.zeros((self.n_clusters, self.kmeans.cluster_centers_.shape[1]))

        # Iterate over all training files to find closest features
        train_files = os.listdir(self.train_dir)
        for file in tqdm(train_files, desc="Finding closest features"):
            if file.endswith(".npz"):
                f = np.load(os.path.join(self.train_dir, file), allow_pickle=True)
                spectogram = f['s'].T  # Transpose to match k-means input
                
                # Calculate distances from each slice to each centroid
                distances = self.kmeans.transform(spectogram)

                # Find the closest slice for each centroid
                for i in range(self.n_clusters):
                    closest_idx = np.argmin(distances[:, i])
                    closest_features[i] = spectogram[closest_idx]

        # Save the closest features to a numpy file
        np.save(save_path, closest_features)
        print(f"Closest features to centroids saved at {save_path}")

    def generate_train_test_labels(self, logits=False):
        # Handle training data
        train_files = [file for file in os.listdir(self.train_dir) if file.endswith(".npz")]
        total_train_files = len(train_files)
        print(f"Processing {total_train_files} training files...")
        
        for i, file in enumerate(train_files):
            self.process_file(file, self.train_dir, logits=logits)
            print(f"Processed training file {i + 1}/{total_train_files}")

        # Handle test data
        test_files = [file for file in os.listdir(self.test_dir) if file.endswith(".npz")]
        total_test_files = len(test_files)
        print(f"Processing {total_test_files} test files...")
        
        for i, file in enumerate(test_files):
            self.process_file(file, self.test_dir, logits=logits)
            print(f"Processed test file {i + 1}/{total_test_files}")

    def process_file(self, file, directory, logits=False):
        f = np.load(os.path.join(directory, file), allow_pickle=True)

        if logits == True:
            spectogram = f['logits'].T

        else:
            spectogram = f['s'].T

        original_spectogram = f['s']
     
        # Predict labels
        labels = self.kmeans.predict(spectogram)

        f_dict = {'s': original_spectogram, 'labels': f['labels'], 'new_labels': labels}
        save_path = os.path.join(directory, file)
        np.savez(save_path, **f_dict)


def iterate_training(model, device, train_dir, test_dir, iteration_n=1):
    # Create directories if they don't exist
    train_iteration = f"train_iteration_{iteration_n}"  # Modified
    test_iteration = f"test_iteration_{iteration_n}"    # Modified

    if not os.path.exists(train_iteration):
        os.mkdir(train_iteration)

    if not os.path.exists(test_iteration):
        os.mkdir(test_iteration)

    # For each file in train and test, pass through the model, save in a new dir, save the logits in place of spectogram
    for file in tqdm(os.listdir(train_dir), desc=f"Processing Train Files for Iteration {iteration_n}"):
        if file.endswith(".npz"):
            f = np.load(os.path.join(train_dir, file))
            spectogram = f['s']
            # Normalize (Z-score normalization)
            mean = spectogram.mean()
            std = spectogram.std()
            spectogram = (spectogram - mean) / (std + 1e-7)
            # Convert to torch tensors
            spectogram = torch.from_numpy(spectogram).float().unsqueeze(0)
            spectogram = spectogram.unsqueeze(0)
            output, _ = model.inference_forward(spectogram.to(device))
            output = output.squeeze(0).T

            f_dict = {'s': f['s'], 'labels': f['labels'], 'new_labels': f['new_labels'], 'logits': output.detach().cpu().numpy()}
            save_path = os.path.join(train_iteration, file)
            np.savez(save_path, **f_dict)

    for file in tqdm(os.listdir(test_dir), desc=f"Processing Test Files for Iteration {iteration_n}"):
        if file.endswith(".npz"):
            f = np.load(os.path.join(test_dir, file))
            spectogram = f['s']
            # Normalize (Z-score normalization)
            mean = spectogram.mean()
            std = spectogram.std()
            spectogram = (spectogram - mean) / (std + 1e-7)
            # Convert to torch tensors
            spectogram = torch.from_numpy(spectogram).float().unsqueeze(0)
            spectogram = spectogram.unsqueeze(0)
            output, _ = model.inference_forward(spectogram.to(device))
            output = output.squeeze(0).T

            f_dict = {'s': f['s'], 'labels': f['labels'], 'new_labels': f['new_labels'], 'logits': output.detach().cpu().numpy()}
            save_path = os.path.join(test_iteration, file)
            np.savez(save_path, **f_dict)