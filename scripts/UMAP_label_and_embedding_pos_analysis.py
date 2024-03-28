import numpy as np
import matplotlib.pyplot as plt
import random

def plot_label_examples(file_path, num_labels, num_examples, spec_length=250):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]  # Spectrogram data
    labels = data["hdbscan_labels"]  # Integer labels per timepoint
    embedding = data["embedding_outputs"]  # Embedding outputs
    
    # Normalize the embedding dimensions between 0 and 1 for the entire dataset
    embedding_normalized = (embedding - embedding.min(axis=0)) / (embedding.max(axis=0) - embedding.min(axis=0))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Randomly select the specified number of labels
    selected_labels = random.sample(list(unique_labels), num_labels)
    
    # Set up the figure and grid
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(num_labels * 2, num_examples, height_ratios=[20, 1] * num_labels)
    
    for i, label in enumerate(selected_labels):
        # Find all occurrences of the current label
        label_indices = np.where(labels == label)[0]
        
        # Randomly select the specified number of examples
        if len(label_indices) >= num_examples:
            example_indices = random.sample(list(label_indices), num_examples)
        else:
            example_indices = label_indices
        
        for j, idx in enumerate(example_indices):
            # Extract the spectrogram slice for the current example
            start_idx = max(0, idx - spec_length // 2)
            end_idx = min(start_idx + spec_length, spec.shape[0])
            spec_slice = spec[start_idx:end_idx, :].T
            
            # Pad or truncate the spectrogram slice to the desired length
            if spec_slice.shape[1] < spec_length:
                pad_width = spec_length - spec_slice.shape[1]
                spec_slice = np.pad(spec_slice, ((0, 0), (0, pad_width)), mode='constant')
            else:
                spec_slice = spec_slice[:, :spec_length]
            
            # Normalize the spectrogram values to the range [0, 1]
            spec_slice_normalized = (spec_slice - np.min(spec_slice)) / (np.max(spec_slice) - np.min(spec_slice))
            
            # Plot the spectrogram slice
            ax_spec = fig.add_subplot(gs[i * 2, j])
            ax_spec.imshow(spec_slice_normalized, aspect='auto', origin='lower', cmap='inferno')
            ax_spec.set_title(f'Label {label}, Example {j+1}')
            ax_spec.axis('off')
            
            # Extract the corresponding embedding slice from the normalized embedding
            embedding_slice_normalized = embedding_normalized[start_idx:end_idx, :]
            
            # Plot the embedding gradient colorbar
            embedding_gradient = np.mean(embedding_slice_normalized, axis=1)
            embedding_gradient = np.tile(embedding_gradient, (2, 1))  # Adjust the height of the colorbar
            ax_emb = fig.add_subplot(gs[i * 2 + 1, j])
            ax_emb.imshow(embedding_gradient, aspect='auto', cmap='viridis', extent=[0, spec_length, 0, 1])
            ax_emb.axis('off')
    
    plt.tight_layout()
    plt.show()

# Load the NPZ file and call the function to plot
file_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_noise_aug_cvm-attn-1-brownthrasher.npz"

# Get user input for the number of labels and examples to visualize
num_labels = 3
num_examples = 3

plot_label_examples(file_path, num_labels, num_examples)