import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    embedding = data["embedding_outputs"]
    ground_truth_labels = data["ground_truth_labels"]
    ground_truth_colors = data["ground_truth_colors"].item()
    ground_truth_colors[int(1)] = "#000000"  # Add black to dictionary with key 0
    hdbscan_labels = data["hdbscan_labels"]
    return embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels

def create_animated_gif(embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels, output_path, points_per_frame=20):
    # Normalize the embedding coordinates
    x_coords = embedding[:, 0]
    y_coords = embedding[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_norm = (x_coords - x_min) / (x_max - x_min)
    y_norm = (y_coords - y_min) / (y_max - y_min)

    # Create a figure and subplots for the animation
    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))

    # Get unique ground truth labels
    unique_labels = np.unique(ground_truth_labels)

    # Initialize the scatter plot with the ground truth colors
    for label in unique_labels:
        mask = ground_truth_labels == label
        color = ground_truth_colors[label]
        ax_anim.scatter(x_norm[mask], y_norm[mask], s=40, alpha=0.5, color=color, label=f'Ground Truth Label {label}')

    ax_anim.set_aspect('equal')
    ax_anim.set_title('Single Song UMAP Embedding', fontsize=24)
    ax_anim.set_xlabel('UMAP 1', fontsize=24)
    ax_anim.set_ylabel('UMAP 2', fontsize=24)

    # Define the animation function
    def animate(frame):
        start_idx = frame * points_per_frame
        end_idx = min(start_idx + points_per_frame, len(embedding))
        colors = np.array(['black'] * len(embedding))
        sizes = np.ones(len(embedding)) * 2

        colors[start_idx:end_idx] = [ground_truth_colors[label] for label in ground_truth_labels[start_idx:end_idx]]
        sizes[start_idx:end_idx] = 80

        ax_anim.clear()
        for label in unique_labels:
            mask = ground_truth_labels == label
            color = ground_truth_colors[label]
            ax_anim.scatter(x_norm[mask], y_norm[mask], s=sizes[mask], alpha=0.5, color=color, label=f'Ground Truth Label {label}')

        ax_anim.set_aspect('equal')
        ax_anim.set_title('Single Song UMAP Embedding', fontsize=24)
        ax_anim.set_xlabel('UMAP 1', fontsize=24)
        ax_anim.set_ylabel('UMAP 2', fontsize=24)

    # Set the number of frames for the animation
    num_frames = len(embedding) // points_per_frame + 1

    # Create the animation
    anim = animation.FuncAnimation(fig_anim, animate, frames=num_frames, interval=50)

    # Save the animation as a GIF
    anim.save(output_path, writer='pillow')

    plt.close(fig_anim)

if __name__ == "__main__":
    file_path = "files/labels_Single_Song2.npz"
    output_path = "animated_dots.gif"
    points_per_frame = 500

    embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels = load_data(file_path)
    create_animated_gif(embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels, output_path, points_per_frame)