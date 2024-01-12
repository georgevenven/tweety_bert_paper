import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=196, remove_silences=False, psuedo_labels_generated=True):
        self.file_path = []
        self.num_classes = num_classes
        self.remove_silences = remove_silences
        self.psuedo_labels_generated = psuedo_labels_generated

        for file in os.listdir(file_dir):
            self.file_path.append(os.path.join(file_dir, file))

    def __getitem__(self, index):
        file_path = self.file_path[index]

        data = np.load(file_path, allow_pickle=True)
        spectogram = data['s']

        ground_truth_labels = data['labels']

        # for the cases when this dataclass is used on datasets that have not had psuedo labels generated for them
        # such an example may be in the eval process 
        if self.psuedo_labels_generated == True:
            psuedo_labels = data['new_labels']
        else:
            psuedo_labels = data['labels']
            # this is sloppy, but essentially the psuedo label generation process also crop the freq dim
            # so I have to do it here if the psuedo label generation process has not occured yet 

            # # Z-score normalization, because this also done in psuedo label generation process 
            # mean_val = spectogram.mean()
            # std_val = spectogram.std()
            # spectogram = (spectogram - mean_val) / (std_val + 1e-7)  # Adding a small constant to prevent division by zero
            # # Replace NaN values with zeros
            # spectogram[np.isnan(spectogram)] = 0
            # spectogram = spectogram[20:216,:]


        max_indices = np.argmax(spectogram, axis=0, keepdims=True)
        max_indices = torch.tensor(max_indices, dtype=torch.int64).squeeze(0)

        # ground truth and psuedo labels is just length vector 
        ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.int64).squeeze(0)
        psuedo_labels = torch.tensor(psuedo_labels, dtype=torch.int64).squeeze(0)

        # bring spectogram to length, height 
        spectogram = torch.from_numpy(spectogram).float()
        spectogram = spectogram.permute(1,0)

        if self.remove_silences == True:
            # find where not equal to 0 
            not_silent_indexes = torch.where(max_indices != 0) 
            # untuple 
            not_silent_indexes = not_silent_indexes[0]
            # keep only those indexes 
            spectogram = spectogram[not_silent_indexes]
            psuedo_labels = psuedo_labels[not_silent_indexes]
            ground_truth_labels = ground_truth_labels[not_silent_indexes]

        # Convert label and psuedo label to one-hot encoding
        ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()
        psuedo_labels = F.one_hot(psuedo_labels, num_classes=self.num_classes).float()

        return spectogram, psuedo_labels, ground_truth_labels

    def __len__(self):
        return len(self.file_path)

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
