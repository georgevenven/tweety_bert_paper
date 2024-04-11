import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import random
import matplotlib.pyplot as plt
import logging

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=40, infinite_loader=True):
        self.file_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir)]
        self.num_classes = num_classes
        self.infinite_loader = infinite_loader

    def __getitem__(self, idx):
        if self.infinite_loader:
            original_idx = idx  # Store the original index for logging
            idx = random.randint(0, len(self.file_paths) - 1)
        file_path = self.file_paths[idx]

        try:
            # Load data and preprocess
            data = np.load(file_path, allow_pickle=True)
            spectogram = data['s']

            spectogram = spectogram[20:216, :]

            # # # Calculate mean and standard deviation of the spectrogram
            spec_mean = np.mean(spectogram)
            spec_std = np.std(spectogram)
            # Z-score the spectrogram
            spectogram = (spectogram - spec_mean) / spec_std

            # # Process labels
            ground_truth_labels = np.array(data['labels'], dtype=int)
            # # else:
            #     # If 'labels' is None or not present, assign a default value or handle it accordingly
            # ground_truth_labels = np.zeros(spectogram.shape[1], dtype=int)

            ground_truth_labels = torch.from_numpy(ground_truth_labels).long().squeeze(0)
            spectogram = torch.from_numpy(spectogram).float().permute(1, 0)
            ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()

            return spectogram, ground_truth_labels

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            # Recursively call __getitem__ with a different index if in infinite loader mode
            if self.infinite_loader:
                return self.__getitem__(random.randint(0, len(self.file_paths) - 1))
            else:
                raise e
    
    def __len__(self):
        if self.infinite_loader:
            # Return an arbitrarily large number to simulate an infinite dataset
            return int(1e12)
        else:
            return len(self.file_paths)

class CollateFunction:
    def __init__(self, segment_length=1000):
        self.segment_length = segment_length

    def __call__(self, batch):
        # with torch.profiler.record_function("collate_fn"):

        # with record_function("collate_fn"):
        # Unzip the batch (a list of (spectogram, psuedo_labels, ground_truth_labels) tuples)
        spectograms, ground_truth_labels = zip(*batch)
        # Create lists to hold the processed tensors
        spectograms_processed = []
        ground_truth_labels_processed = []

        # Each sample in batch
        for spectogram, ground_truth_label in zip(spectograms, ground_truth_labels):
            # Truncate if larger than context window
            if spectogram.shape[0] > self.segment_length:
                # get random view of size segment
                # find range of valid starting pts (essentially these are the possible starting pts for the length to equal segment window)
                starting_points_range = spectogram.shape[0] - self.segment_length        
                start = torch.randint(0, starting_points_range, (1,)).item()  
                end = start + self.segment_length     

                spectogram = spectogram[start:end]
                ground_truth_label = ground_truth_label[start:end]

            # Pad with 0s if shorter
            if spectogram.shape[0] < self.segment_length:
                pad_amount = self.segment_length - spectogram.shape[0]
                spectogram = F.pad(spectogram, (0, 0, 0, pad_amount), 'constant', 0)
                ground_truth_label = F.pad(ground_truth_label, (0, 0, 0, pad_amount), 'constant', 0)  # Adjusted padding for labels

            # Append the processed tensors to the lists
            spectograms_processed.append(spectogram)
            ground_truth_labels_processed.append(ground_truth_label)

        # Stack tensors along a new dimension to match the BERT input size.
        spectograms = torch.stack(spectograms_processed, dim=0)
        ground_truth_labels = torch.stack(ground_truth_labels_processed, dim=0)

        # Final reshape for model
        spectograms = spectograms.unsqueeze(1).permute(0,1,3,2)

        return spectograms, ground_truth_labels


# # Initialize your dataset and collate_fn
# dataset = SongDataSet_Image(file_dir='files/yarden_test')
# collate_fn = CollateFunction(segment_length=1000)

# # Create the DataLoader
# data_loader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=collate_fn, num_workers=16)

# # Start profiling
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     for i, (spectograms, ground_truth_labels) in enumerate(data_loader):
#         if i >= 50:  
#             break

# # Print profiler results
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))


