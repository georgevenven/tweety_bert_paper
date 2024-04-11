

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
        max_length = 500
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
    ground_truth_colors = data["ground_truth_colors"].item()
    ground_truth_colors[int(1)] = "#000000"  # Add black to dictionary with key 0

    print(ground_truth_colors)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get unique ground truth labels
    unique_labels = np.unique(ground_truth_labels)

    # Plot points with colors based on ground truth labels
    for label in unique_labels:
        mask = ground_truth_labels == label
        color = ground_truth_colors[label]
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=2, alpha=0.1, color=color, label=f'Ground Truth Label {label}')

    lasso = LassoSelector(ax, onselect)

    ax.set_title('UMAP Projection')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    plt.show()

file_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_LLB11_Yarden_FreqTruncated_Trained_Model_attention-2_500k.npz"
plot_umap_with_selection(file_path)

