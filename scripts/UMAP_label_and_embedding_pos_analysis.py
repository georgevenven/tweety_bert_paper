import numpy as np
import matplotlib.pyplot as plt
import random

def plot_label_examples(file_path, num_labels, num_examples, spec_length=100):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]  # Spectrogram data
    labels = data["hdbscan_labels"]  # Integer labels per timepoint
    embedding = data["embedding_outputs"]  # UMAP embedding positions
    
    # Normalize the embedding positions between 0 and 1 for the whole array before processing
    embedding_min = embedding.min(axis=0)
    embedding_max = embedding.max(axis=0)
    embedding_normalized = (embedding - embedding_min) / (embedding_max - embedding_min)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Randomly select the specified number of labels
    selected_labels = random.sample(list(unique_labels), num_labels)
    
    # Calculate dynamic figure size
    plot_height_ratio = 9  # This will be the ratio for the spectrogram plots
    bar_height_ratio = 1   # This will be the ratio for the embedding bars, 10% of the spectrogram plot height
    width_per_example = 4  # Adjust this value to change the width of each example

    # Calculate total figure height based on the ratios
    total_height = (plot_height_ratio + bar_height_ratio) * num_labels
    total_width = width_per_example * num_examples

    # Set up the figure with the calculated size
    fig = plt.figure(figsize=(total_width, total_height))

    # Adjust the height ratios for the grid specification
    height_ratios = [plot_height_ratio if i % 2 == 0 else bar_height_ratio for i in range(num_labels * 2)]
    gs = fig.add_gridspec(num_labels * 2, num_examples, height_ratios=height_ratios)
    
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
        
        # Initialize a list to keep track of used indices to ensure no overlap
        used_indices = []
        
        # Process each selected sequence
        for j in range(min(num_examples, len(sequences))):
            seq = sequences[j]
            seq_length = seq[1] - seq[0]
            
            # Calculate the midpoint of the sequence
            mid_point = seq[0] + seq_length // 2
            
            # Adjust start and end indices based on the spec_length
            start_idx = max(0, mid_point - spec_length // 2)
            end_idx = start_idx + spec_length
            
            # Check if the current indices overlap with any used indices
            if any(start < end_idx and end > start_idx for start, end in used_indices):
                continue  # Skip this sequence if there's an overlap
            
            # Update used indices with the current sequence's indices
            used_indices.append((start_idx, end_idx))
            
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
            ax_spec = fig.add_subplot(gs[i * 2, j])
            ax_spec.imshow(spec_slice_normalized, aspect='auto', origin='lower', cmap='inferno')
            
            # Normalize each dimension of the embedding slice to [0, 1] for visualization
            embedding_slice = embedding_normalized[start_idx:end_idx]
            embedding_slice_norm = (embedding_slice - embedding_slice.min(axis=0)) / (embedding_slice.max(axis=0) - embedding_slice.min(axis=0))

            # Use the first dimension to determine the intensity of the yellow color
            yellow_intensity = embedding_slice_norm[:, 0]
            # Use the second dimension to determine the intensity of the green color
            green_intensity = embedding_slice_norm[:, 1]

            # Create the color blend for each bar
            colors = np.zeros((spec_length, 4))  # Initialize an array for RGBA colors
            colors[:, 0] = 1.0 * yellow_intensity  # Red channel, more intensity means more yellow
            colors[:, 1] = 1.0 * green_intensity   # Green channel, more intensity means more green
            colors[:, 2] = 0.0                     # No blue channel
            colors[:, 3] = 1.0                     # Alpha channel set to fully opaque

            # Now, visualize this as a bar plot over the spectrogram or as a separate visualization
            ax_embedding = fig.add_subplot(gs[i * 2 + 1, j])  # Adjust grid spec index as needed

            # Create a bar plot where each bar represents an index in the spectrogram slice
            # The position on the x-axis corresponds to the index, and the y-axis is fixed
            # The color of each bar is determined by the embedding dimension mapped above
            x_positions = np.arange(spec_length)  # Generate x positions for each embedding value
            bar_width = 1.0  # Width of the bars, can be adjusted as needed

            # Create the bar plot
            for pos, color in zip(x_positions, colors):
                ax_embedding.bar(pos, 1, width=bar_width, color=color, align='edge')

            ax_embedding.set_xlim(0, spec_length)  # Set the x-axis limits to match the spectrogram's x-axis
            ax_embedding.axis('off')  # Hide axes for cleaner visualization

    plt.tight_layout()
    plt.show()

# Load the NPZ file and call the function
file_path = "files/labels_bengalese_attn-3.npz"

num_labels = 3
num_examples = 3
plot_label_examples(file_path, num_labels, num_examples)