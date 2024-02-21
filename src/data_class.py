import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=196, remove_silences=False, psuedo_labels_generated=True, max_retries = 100):
        self.file_paths = []
        self.num_classes = num_classes
        self.remove_silences = remove_silences
        self.pseudo_labels_generated = psuedo_labels_generated
        self.max_retries = max_retries

        for file in os.listdir(file_dir):
            self.file_paths.append(os.path.join(file_dir, file))

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        data = np.load(file_path, allow_pickle=True)
        spectogram = data['s']

        ground_truth_labels = data['labels']

        if self.pseudo_labels_generated:
            pseudo_labels = data['new_labels']
        else:
            pseudo_labels = data['labels']
            # Z-score normalization
            mean_val, std_val = spectogram.mean(), spectogram.std()
            spectogram = (spectogram - mean_val) / (std_val + 1e-7)
            spectogram[np.isnan(spectogram)] = 0
            spectogram = spectogram[20:216, :]

        max_indices = np.argmax(spectogram, axis=0, keepdims=True)
        max_indices = torch.tensor(max_indices, dtype=torch.int64).squeeze(0)

        ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.int64).squeeze(0)
        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.int64).squeeze(0)

        spectogram = torch.from_numpy(spectogram).float().permute(1, 0)

        if self.remove_silences:
            not_silent_indexes = torch.where(max_indices != 0)[0]
            spectogram = spectogram[not_silent_indexes]
            pseudo_labels = pseudo_labels[not_silent_indexes]
            ground_truth_labels = ground_truth_labels[not_silent_indexes]

        ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()
        pseudo_labels = F.one_hot(pseudo_labels, num_classes=self.num_classes).float()

        return spectogram, pseudo_labels, ground_truth_labels

    def __len__(self):
        return len(self.file_paths)

class CollateFunction:
    def __init__(self, segment_length=1000):
        self.segment_length = segment_length

    def __call__(self, batch):
        # Unzip the batch (a list of (spectogram, psuedo_labels, ground_truth_labels) tuples)
        spectograms, psuedo_labels, ground_truth_labels = zip(*batch)

        # Create lists to hold the processed tensors
        spectograms_processed = []
        psuedo_labels_processed = []
        ground_truth_labels_processed = []

        # Each sample in batch
        for spectogram, psuedo_label, ground_truth_label in zip(spectograms, psuedo_labels, ground_truth_labels):

            # Truncate if larger than context window
            if spectogram.shape[0] > self.segment_length:
                # get random view of size segment
                # find range of valid starting pts (essentially these are the possible starting pts for the length to equal segment window)
                starting_points_range = spectogram.shape[0] - self.segment_length        
                start = torch.randint(0, starting_points_range, (1,)).item()  
                end = start + self.segment_length     

                spectogram = spectogram[start:end]
                psuedo_label = psuedo_label[start:end]
                ground_truth_label = ground_truth_label[start:end]

            # Pad with 0s if shorter
            if spectogram.shape[0] < self.segment_length:
                pad_amount = self.segment_length - spectogram.shape[0]
                spectogram = F.pad(spectogram, (0, 0, 0, pad_amount), 'constant', 0)
                psuedo_label = F.pad(psuedo_label, (0, 0, 0, pad_amount), 'constant', 0)  # Adjusted padding for labels
                ground_truth_label = F.pad(ground_truth_label, (0, 0, 0, pad_amount), 'constant', 0)  # Adjusted padding for labels

            # Append the processed tensors to the lists
            spectograms_processed.append(spectogram)
            psuedo_labels_processed.append(psuedo_label)
            ground_truth_labels_processed.append(ground_truth_label)

        # Stack tensors along a new dimension to match the BERT input size.
        # You might need to adjust dimensions based on your exact use case.
        spectograms = torch.stack(spectograms_processed, dim=0)
        psuedo_labels = torch.stack(psuedo_labels_processed, dim=0)
        ground_truth_labels = torch.stack(ground_truth_labels_processed, dim=0)

        # Final reshape for model
        spectograms = spectograms.unsqueeze(1).permute(0,1,3,2)

        return spectograms, psuedo_labels, ground_truth_labels
