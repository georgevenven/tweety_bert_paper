import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

def load_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    embedding = data["embedding_outputs"]
    ground_truth_labels = data["ground_truth_labels"]
    ground_truth_colors = data["ground_truth_colors"].item()
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

    # Create a custom colormap for HDBSCAN labels
    unique_labels = np.unique(hdbscan_labels)
    num_labels = len(unique_labels)
    cmap_colors = plt.cm.get_cmap('viridis', num_labels)
    cmap_hdbscan = mcolors.ListedColormap(cmap_colors(np.linspace(0, 1, num_labels)))

    # Initialize the scatter plot with the initial colors
    scatter = ax_anim.scatter(x_norm, y_norm, s=70, c=hdbscan_labels, alpha=0.1, cmap=cmap_hdbscan)
    ax_anim.set_aspect('equal')
    ax_anim.set_title('Animated Dots', fontsize=18)
    ax_anim.set_xlabel('Normalized X', fontsize=14)
    ax_anim.set_ylabel('Normalized Y', fontsize=14)
    ax_anim.tick_params(axis='both', which='major', labelsize=12)

    # Remove tick marks and labels
    ax_anim.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Set the spine properties
    for spine in ax_anim.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)

    # Define the animation function
    def animate(frame):
        start_idx = frame * points_per_frame
        end_idx = min(start_idx + points_per_frame, len(embedding))
        colors = np.array(['black'] * len(embedding))
        sizes = np.ones(len(embedding)) * 70

        colors[start_idx:end_idx] = [ground_truth_colors[label] for label in ground_truth_labels[start_idx:end_idx]]
        sizes[start_idx:end_idx] = 100

        scatter.set_color(colors)
        scatter.set_sizes(sizes)
        scatter.set_edgecolors(['white' if i >= start_idx and i < end_idx else 'none' for i in range(len(embedding))])

        return scatter,

    # Set the number of frames for the animation
    num_frames = len(embedding) // points_per_frame + 1

    # Create the animation
    anim = animation.FuncAnimation(fig_anim, animate, frames=num_frames, interval=50, blit=True)

    # Save the animation as a GIF
    anim.save(output_path, writer='pillow')

    plt.close(fig_anim)

if __name__ == "__main__":
    file_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_Single_Song.npz"
    output_path = "animated_dots.gif"
    points_per_frame = 500

    embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels = load_data(file_path)
    create_animated_gif(embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels, output_path, points_per_frame)