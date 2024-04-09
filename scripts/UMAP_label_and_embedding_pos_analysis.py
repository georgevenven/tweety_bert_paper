# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import glasbey

# def plot_label_examples_with_umap_highlight(file_path, selected_labels, num_examples, spec_length=500):
#     # Load data from the .npz file
#     data = np.load(file_path, allow_pickle=True)
#     spec = data["s"]  # Spectrogram data
#     labels = data["ground_truth_labels"]  # Integer labels per timepoint
#     embedding = data["embedding_outputs"]  # UMAP embedding positions
    
#     # Normalize the embedding positions between 0 and 1 for the whole array before processing
#     embedding_min = embedding.min(axis=0)
#     embedding_max = embedding.max(axis=0)
#     embedding_normalized = (embedding - embedding_min) / (embedding_max - embedding_min)
    
#     # Ensure selected_labels is a list
#     if not isinstance(selected_labels, list):
#         raise ValueError("selected_labels must be a list of integers.")
    
#     # Filter out any labels not in the dataset to avoid errors
#     unique_labels = np.unique(labels)
#     selected_labels = [label for label in selected_labels if label in unique_labels]
    
#     # Calculate dynamic figure size
#     num_labels = len(selected_labels)  # Update num_labels to reflect the length of selected_labels
#     plot_height_ratio = 9  # This will be the ratio for the spectrogram plots
#     bar_height_ratio = 1   # This will be the ratio for the embedding bars, 10% of the spectrogram plot height
#     width_per_example = 4  # Adjust this value to change the width of each example

#     # Calculate total figure height based on the ratios
#     total_height = (plot_height_ratio + bar_height_ratio) * num_labels
#     total_width = width_per_example * num_examples

#     # Set up the figure with the calculated size
#     fig = plt.figure(figsize=(total_width, total_height))

#     # Adjust the height ratios for the grid specification
#     height_ratios = [plot_height_ratio if i % 2 == 0 else bar_height_ratio for i in range(num_labels * 2)]
#     gs = fig.add_gridspec(num_labels * 2, num_examples, height_ratios=height_ratios)
    
#     for i, label in enumerate(selected_labels):
#         # Find all occurrences of the current label
#         label_indices = np.where(labels == label)[0]
        
#         # Find all continuous sequences of the current label
#         sequences = []
#         seq_start = label_indices[0]
#         for j in range(1, len(label_indices)):
#             if label_indices[j] != label_indices[j-1] + 1:
#                 sequences.append((seq_start, label_indices[j-1] + 1))
#                 seq_start = label_indices[j]
#         sequences.append((seq_start, label_indices[-1] + 1))  # Add the last sequence
        
#         # Sort sequences by their length in descending order
#         sequences.sort(key=lambda x: x[1] - x[0], reverse=True)
        
#         # Initialize a list to keep track of used indices to ensure no overlap
#         used_indices = []
        
#         # Process each selected sequence
#         for j in range(min(num_examples, len(sequences))):
#             seq = sequences[j]
#             seq_length = seq[1] - seq[0]
            
#             # Calculate the midpoint of the sequence
#             mid_point = seq[0] + seq_length // 2
            
#             # Adjust start and end indices based on the spec_length
#             start_idx = max(0, mid_point - spec_length // 2)
#             end_idx = start_idx + spec_length
            
#             # Check if the current indices overlap with any used indices
#             if any(start < end_idx and end > start_idx for start, end in used_indices):
#                 continue  # Skip this sequence if there's an overlap
            
#             # Update used indices with the current sequence's indices
#             used_indices.append((start_idx, end_idx))
            
#             # Ensure the end index does not exceed the bounds of the spectrogram data
#             if end_idx > spec.shape[0]:
#                 end_idx = spec.shape[0]
#                 start_idx = max(0, end_idx - spec_length)
            
#             # Extract the spectrogram slice
#             spec_slice = spec[start_idx:end_idx, :].T
            
#             # Pad the spectrogram slice to the desired length if necessary
#             if spec_slice.shape[1] < spec_length:
#                 pad_width = spec_length - spec_slice.shape[1]
#                 spec_slice = np.pad(spec_slice, ((0, 0), (0, pad_width)), mode='constant')
            
#             # Normalize the spectrogram values to the range [0, 1]
#             spec_slice_normalized = (spec_slice - np.min(spec_slice)) / (np.max(spec_slice) - np.min(spec_slice))
            
#             # Plot the spectrogram slice
#             ax_spec = fig.add_subplot(gs[i * 2, j])
#             ax_spec.imshow(spec_slice_normalized, aspect='auto', origin='lower', cmap='inferno')
            
#             # Normalize each dimension of the embedding slice to [0, 1] for visualization
#             embedding_slice = embedding_normalized[start_idx:end_idx]
#             embedding_slice_norm = (embedding_slice - embedding_slice.min(axis=0)) / (embedding_slice.max(axis=0) - embedding_slice.min(axis=0))

#             # Use the first dimension to determine the intensity of the yellow color
#             yellow_intensity = embedding_slice_norm[:, 0]
#             # Use the second dimension to determine the intensity of the green color
#             green_intensity = embedding_slice_norm[:, 1]

#             # Create the color blend for each bar
#             colors = np.zeros((spec_length, 4))  # Initialize an array for RGBA colors
#             colors[:, 0] = 1.0 * yellow_intensity  # Red channel, more intensity means more yellow
#             colors[:, 1] = 1.0 * green_intensity   # Green channel, more intensity means more green
#             colors[:, 2] = 0.0                     # No blue channel
#             colors[:, 3] = 1.0                     # Alpha channel set to fully opaque

#             # Now, visualize this as a bar plot over the spectrogram or as a separate visualization
#             ax_embedding = fig.add_subplot(gs[i * 2 + 1, j])  # Adjust grid spec index as needed

#             # Create a bar plot where each bar represents an index in the spectrogram slice
#             # The position on the x-axis corresponds to the index, and the y-axis is fixed
#             # The color of each bar is determined by the embedding dimension mapped above
#             x_positions = np.arange(spec_length)  # Generate x positions for each embedding value
#             bar_width = 1.0  # Width of the bars, can be adjusted as needed

#             # Create the bar plot
#             for pos, color in zip(x_positions, colors):
#                 ax_embedding.bar(pos, 1, width=bar_width, color=color, align='edge')

#             ax_embedding.set_xlim(0, spec_length)  # Set the x-axis limits to match the spectrogram's x-axis
#             ax_embedding.axis('off')  # Hide axes for cleaner visualization

#     plt.tight_layout()
#     plt.show()

#     # After plotting spectrograms, plot the UMAP with highlighted selected labels
#     plt.figure(figsize=(8, 6))
#     # Plot all points as a background
#     plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5, label='All Points')
    
#     # Highlight the selected labels
#     for label in selected_labels:
#         # Find indices of the current label
#         indices = np.where(labels == label)[0]
#         # Extract the corresponding embedding points
#         selected_embedding = embedding[indices]
#         # Plot these points with a different style
#         plt.scatter(selected_embedding[:, 0], selected_embedding[:, 1], s=10, alpha=0.75, label=f'Label {label}')
    
#     plt.legend()
#     plt.title('UMAP Projection with Highlighted Labels')
#     plt.xlabel('UMAP 1')
#     plt.ylabel('UMAP 2')
#     plt.show()

# # import numpy as np

# # # Loading the NPZ file to access its contents
# file_path = "files/labels_LLB3_Yarden_Trained_Model_attention-1,5k_test.npz"
# # data = np.load(file_path, allow_pickle=True)
# # ground_truth_colors = data["ground_truth_colors"].item()  # Assuming ground_truth_colors is stored as a dictionary in the NPZ file
# # # Plot each ground truth color with the associated number
# # plt.figure(figsize=(10, 2))
# # for i, (number, color) in enumerate(ground_truth_colors.items()):
# #     plt.fill_between([i, i+1], 0, 1, color=color, edgecolor='black')
# #     plt.text(i+0.5, 0.5, str(number), ha='center', va='center', color='white')
# # plt.xlim(0, len(ground_truth_colors))
# # plt.axis('off')
# # plt.show()

# selected_labels = [3,4]  # Example labels you want to visualize
# num_examples = 2
# plot_label_examples_with_umap_highlight(file_path, selected_labels, num_examples)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import os
import random
import string

def find_longest_contiguous_region(points, labels):
    longest_region = []
    current_region = []
    current_label = None

    for i, point in enumerate(points):
        if labels[point] != current_label:
            if len(current_region) > len(longest_region):
                longest_region = current_region
            current_region = [point]
            current_label = labels[point]
        else:
            current_region.append(point)

    if len(current_region) > len(longest_region):
        longest_region = current_region

    return longest_region, current_label

def onselect(verts):
    global embedding, spec, labels, ground_truth_labels

    path = Path(verts)
    mask = path.contains_points(embedding)
    selected_points = np.where(mask)[0]

    if len(selected_points) > 0:
        region, hdbscan_label = find_longest_contiguous_region(selected_points, labels)

        # Truncate the spectrogram region if it is longer than 1000
        max_length = 1000
        if len(region) > max_length:
            region = region[:max_length]

        spec_region = spec[region]
        ground_truth_label = ground_truth_labels[region[0]]

        # Generate a random name
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        fig, ax = plt.subplots(figsize=(12, 6))  # Increase the figure size
        ax.imshow(spec_region.T, aspect='auto', origin='lower', cmap='inferno')
        ax.set_title(f"{random_name} - HDBSCAN Label: {hdbscan_label} - Ground Truth Label: {ground_truth_label}", fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

        os.makedirs("imgs/selected_regions", exist_ok=True)
        fig.savefig(f"imgs/selected_regions/{random_name}_hdbscan_{hdbscan_label}_groundtruth_{ground_truth_label}.png")
        plt.close(fig)

def plot_umap_with_selection(file_path):
    global embedding, spec, labels, ground_truth_labels

    data = np.load(file_path, allow_pickle=True)
    embedding = data["embedding_outputs"]
    spec = data["s"]
    labels = data["hdbscan_labels"]
    ground_truth_labels = data["ground_truth_labels"]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
    lasso = LassoSelector(ax, onselect)

    ax.set_title('UMAP Projection')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    plt.show()

file_path = "files/labels_LLB3_Yarden_Trained_Model_attention-1,5k_test.npz"
plot_umap_with_selection(file_path)