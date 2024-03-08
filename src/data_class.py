import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=196):
        self.file_paths = []
        self.num_classes = num_classes

        for file in os.listdir(file_dir):
            self.file_paths.append(os.path.join(file_dir, file))

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        data = np.load(file_path, allow_pickle=True)
        spectogram = data['s']
        # spectogram = spectogram[20:216]

        # Remove when free from Yarden's Spec Gen Method 
        # Z-score normalization
        mean = spectogram.mean()
        std = spectogram.std()
        spectogram = (spectogram - mean) / std

        # Check if 'labels' key exists in data, if not, create a one-dimensional array
        if 'labels' in data:
            ground_truth_labels = data['labels']
        else:
            ground_truth_labels = np.zeros(spectogram.shape[1], dtype=int)  # all zeros represent the lack of labels

        ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.int64).squeeze(0)
        spectogram = torch.from_numpy(spectogram).float().permute(1, 0)
        ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()

        return spectogram, ground_truth_labels

    def __len__(self):
        return len(self.file_paths)

class CollateFunction:
    def __init__(self, segment_length=1000):
        self.segment_length = segment_length

    def __call__(self, batch):
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
