import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import os
import random
import string

class UMAPSelector:
    def __init__(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.embedding = data["embedding_outputs"]
        self.spec = data["s"]
        self.labels = data["hdbscan_labels"]
        self.ground_truth_labels = data["ground_truth_labels"]
        self.ground_truth_colors = data["ground_truth_colors"].item()
        self.ground_truth_colors[int(1)] = "#000000"  # Add black to dictionary with key 0
        self.selected_points = None
        self.selected_embedding = None

    def find_longest_contiguous_region(self, points, labels):
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

    def onselect(self, verts):
        path = Path(verts)
        mask = path.contains_points(self.embedding)
        self.selected_points = np.where(mask)[0]
        if len(self.selected_points) > 0:
            self.selected_embedding = self.embedding[self.selected_points]
            self.plot_selected_region()

    def plot_umap_with_selection(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(self.ground_truth_labels)
        for label in unique_labels:
            mask = self.ground_truth_labels == label
            color = self.ground_truth_colors[label]
            ax.scatter(self.embedding[mask, 0], self.embedding[mask, 1], s=2, alpha=0.1, color=color, label=f'Ground Truth Label {label}')
        lasso = LassoSelector(ax, self.onselect)
        ax.set_title('UMAP Projection')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.show()

    def plot_selected_region(self):
        if self.selected_points is not None:
            region, hdbscan_label = self.find_longest_contiguous_region(self.selected_points, self.labels)
            max_length = 500
            if len(region) > max_length:
                region = region[:max_length]
            spec_region = self.spec[region]
            ground_truth_label = self.ground_truth_labels[region[0]]
            random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

            # Normalize x and y coordinates
            x_coords = self.selected_embedding[:, 0]
            y_coords = self.selected_embedding[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_norm = (x_coords - x_min) / (x_max - x_min)
            y_norm = (y_coords - y_min) / (y_max - y_min)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [6, 1]})
            ax1.imshow(spec_region.T, aspect='auto', origin='lower', cmap='inferno')
            ax1.set_title(f"{random_name} - HDBSCAN Label: {hdbscan_label} - Ground Truth Label: {ground_truth_label}", fontsize=10)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Frequency')

            # Create color gradient based on normalized x and y coordinates
            color_gradient = np.zeros((1, len(x_norm), 3))
            color_gradient[0, :, 0] = x_norm
            color_gradient[0, :, 1] = y_norm
            ax2.imshow(color_gradient, aspect='auto')
            ax2.set_axis_off()

            os.makedirs("imgs/selected_regions", exist_ok=True)
            fig.savefig(f"imgs/selected_regions/{random_name}_hdbscan_{hdbscan_label}_groundtruth_{ground_truth_label}.png")
            plt.close(fig)

file_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_LLB11_Yarden_FreqTruncated_Trained_Model_attention-2_500k.npz"
selector = UMAPSelector(file_path)
selector.plot_umap_with_selection()