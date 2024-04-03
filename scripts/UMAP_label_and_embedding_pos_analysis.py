import numpy as np
import matplotlib.pyplot as plt
import random

def plot_label_examples(file_path, num_labels, num_examples, spec_length=100):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]  # Spectrogram data
    labels = data["hdbscan_labels"]  # Integer labels per timepoint
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Randomly select the specified number of labels
    selected_labels = random.sample(list(unique_labels), num_labels)
    
    # Set up the figure and grid
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(num_labels, num_examples, height_ratios=[20] * num_labels)
    
    for i, label in enumerate(selected_labels):
        # Find all occurrences of the current label
        label_indices = np.where(labels == label)[0]
        
        # Find all continuous sequences of the current label
        sequences = []
        seq_start = label_indices[0]
        for j in range(1, len(label_indices)):
            if label_indices[j] != label_indices[j-1] + 1:
                sequences.append((seq_start, label_indices[j-1] + 1))
                seq_start = label_indices[j]
        sequences.append((seq_start, label_indices[-1] + 1))  # Add the last sequence
        
        # Sort sequences by their length in descending order
        sequences.sort(key=lambda x: x[1] - x[0], reverse=True)
        
        # Process each selected sequence
        for j in range(min(num_examples, len(sequences))):
            seq = sequences[j]
            seq_length = seq[1] - seq[0]
            
            # Calculate the midpoint of the sequence
            mid_point = seq[0] + seq_length // 2
            
            # Adjust start and end indices based on the spec_length
            start_idx = max(0, mid_point - spec_length // 2)
            end_idx = start_idx + spec_length
            
            # Ensure the end index does not exceed the bounds of the spectrogram data
            if end_idx > spec.shape[0]:
                end_idx = spec.shape[0]
                start_idx = max(0, end_idx - spec_length)
            
            # Extract the spectrogram slice
            spec_slice = spec[start_idx:end_idx, :].T
            
            # Pad the spectrogram slice to the desired length if necessary
            if spec_slice.shape[1] < spec_length:
                pad_width = spec_length - spec_slice.shape[1]
                spec_slice = np.pad(spec_slice, ((0, 0), (0, pad_width)), mode='constant')
            
            # Normalize the spectrogram values to the range [0, 1]
            spec_slice_normalized = (spec_slice - np.min(spec_slice)) / (np.max(spec_slice) - np.min(spec_slice))
            
            # Plot the spectrogram slice
            ax_spec = fig.add_subplot(gs[i, j])
            ax_spec.imshow(spec_slice_normalized, aspect='auto', origin='lower', cmap='inferno')
            # Include the range of indices and class label in the title
            ax_spec.set_title(f'Label {label}, Example {j+1}\nIndices: {start_idx}-{end_idx}')
            ax_spec.axis('off')
    
    plt.tight_layout()
    plt.show()

# Load the NPZ file and call the function to plot
file_path = "files/labels_llb3_attn_3-cosine-nonoise_added.npz"

# Get user input for the number of labels and examples to visualize
num_labels = 3
num_examples = 3

plot_label_examples(file_path, num_labels, num_examples)
