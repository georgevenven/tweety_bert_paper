import pickle
import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_class import SongDataSet_Image, CollateFunction
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler


def load_data( data_dir,remove_silences=False, context=1000, psuedo_labels_generated=False):
    collate_fn = CollateFunction(context)
    dataset = SongDataSet_Image(data_dir, num_classes=196, remove_silences=False, psuedo_labels_generated=psuedo_labels_generated)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=16)
    return loader 

def plot_spectrogram_with_labels(spectrogram, labels):
    """
    Plots a spectrogram with a corresponding line plot overlay representing label values between 0 and 1.

    Args:
        spectrogram (np.array): 2D array representing the spectrogram, shape (time_bins, frequencies).
        labels (List[float]): List of label values for each timebin in the spectrogram, ranging from 0 to 1.
    """

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot the spectrogram
    cax = ax.imshow(spectrogram, aspect='auto', origin='lower')

    # Create time points for x-axis of the line plot
    time_points = np.arange(len(labels))

    # Overlay the line plot on the spectrogram
    # Note: Adjust the line plot's y-values to match the spectrogram's y-axis
    ax.plot(time_points, labels * spectrogram.shape[0], color='cyan', linewidth=2)

    # Add colorbar for the spectrogram
    fig.colorbar(cax, ax=ax, orientation='vertical')

    plt.show()

from sklearn.cluster import OPTICS
import numpy as np

def generate_optics_labels(array, min_samples=10):
    """
    Applies OPTICS clustering to an array of data points and returns the cluster labels.

    Args:
        array (numpy.ndarray): An n x features array of data points.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        numpy.ndarray: An n x 1 array of cluster labels for each data point.
    """

    # Initialize the OPTICS clusterer
    clusterer = OPTICS(min_samples=min_samples, min_cluster_size=250, cluster_method="xi", metric='euclidean')

    # Fit the model to the data and predict cluster labels
    labels = clusterer.fit_predict(array)

    # Reshape labels to n x 1
    labels = labels.reshape(-1, 1)

    print(np.unique(labels))

    return labels

def calculate_relative_position_labels(ground_truth_labels, silence=0):
    labels_array = np.array(ground_truth_labels)
    relative_positions = np.zeros_like(labels_array, dtype=float)

    start_idx = None
    current_label = None

    for i, label in enumerate(labels_array):
        # Check if it's the last element to prevent out-of-bounds access
        is_last_element = i == len(labels_array) - 1

        if label != silence:
            if start_idx is None:
                start_idx = i
                current_label = label
            elif label != current_label:
                phrase_length = i - start_idx
                relative_positions[start_idx:i] = np.linspace(0, 1, phrase_length)
                start_idx = i
                current_label = label
        elif not is_last_element and (labels_array[i + 1] != silence and labels_array[i + 1] != current_label):
            if start_idx is not None:
                end_idx = i if labels_array[i + 1] != current_label else i + 1
                phrase_length = end_idx - start_idx
                relative_positions[start_idx:end_idx] = np.linspace(0, 1, phrase_length)
                start_idx = None

    # Handle the case where the last label is part of an ongoing phrase
    if start_idx is not None:
        phrase_length = len(labels_array) - start_idx
        relative_positions[start_idx:] = np.linspace(0, 1, phrase_length)

    return relative_positions.tolist()

def plot_umap_projection(model, device, data_dir="test_llb16",
                         remove_silences=False, samples=100, file_path='category_colors.pkl', 
                         layer_index=None, dict_key=None, time_bins_per_umap_point=100, 
                         context=1000, save_dir=None, raw_spectogram=False, save_dict_for_analysis=False, compute_svm=False, color_scheme="Syllable"):
    predictions_arr = []
    ground_truth_labels_arr = []
    spec_arr = [] 

    # Reset Figure
    plt.figure(figsize=(8, 6))

    # to allow sci notation 
    samples = int(samples)

    data_loader = load_data(data_dir=data_dir, remove_silences=remove_silences, context=context, psuedo_labels_generated=False)
    data_loader_iter = iter(data_loader)

    while len(ground_truth_labels_arr * context) < samples:
        # Because of the random windows being drawn from songs, it makes sense to reinit dataloader for UMAP plots 
        try:
            # Retrieve the next batch
            data, _, ground_truth_label = next(data_loader_iter)

        except StopIteration:
            # Reinitialize the DataLoader iterator when all batches are exhausted
            data_loader_iter = iter(data_loader)
            data, _, ground_truth_label = next(data_loader_iter)

        if raw_spectogram == False:
            embedding_output, layers = model.inference_forward(data.to(device))

            layer_output_dict = layers[layer_index]
            output = layer_output_dict.get(dict_key, None)

            if output is None:
                print(f"Invalid key: {dict_key}. Skipping this batch.")
                continue

            # number of times the spectogram must be broken apart 
            num_times = context // time_bins_per_umap_point
            batches, time_bins, features = output.shape 

            # data shape [0] is the number of batches, 
            predictions = output.reshape(batches, num_times, time_bins_per_umap_point, features)
            
            # combine the batches and number of samples per context window 
            predictions = predictions.flatten(0,1)
            # combine features and time bins
            predictions = predictions.flatten(-2,-1)
            predictions_arr.append(predictions.detach().cpu().numpy())

        data = data.squeeze(1)
        spec = data

        # set the features (freq axis to be the last dimension)
        spec = spec.permute(0, 2, 1)
        # combine batches and timebins
        spec = spec.flatten(0, 1)

        ground_truth_label = ground_truth_label.flatten(0, 1)

        ground_truth_label = torch.argmax(ground_truth_label, dim=-1)

        spec_arr.append(spec.cpu().numpy())
        ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
        
    with open(file_path, 'rb') as file:
        color_map_data = pickle.load(file)

    label_to_color = {label: tuple(color) for label, color in color_map_data.items()}
    
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
    spec_arr = np.concatenate(spec_arr, axis=0)

    if not raw_spectogram:
        predictions = np.concatenate(predictions_arr, axis=0)
    else:
        predictions = spec_arr

    # razor off any extra datapoints 
    if samples > len(predictions):
        samples = len(predictions)
    else:
        predictions = predictions[:samples]
        ground_truth_labels = ground_truth_labels[:samples]
        
    # Fit the UMAP reducer       
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.05, n_components=2, metric='cosine')

    embedding_outputs = reducer.fit_transform(predictions)
    # hdbscan_labels = generate_optics_labels(predictions)

    if remove_silences == True:
        index_where_silence = np.where(ground_truth_labels != 0) 
        embedding_outputs = embedding_outputs[index_where_silence]
        ground_truth_labels = ground_truth_labels[index_where_silence]

    if compute_svm:
        # Fit SVM on the UMAP embeddings
        scaler = StandardScaler()
        embedding_scaled = scaler.fit_transform(embedding_outputs)
        svm_model = SVC(kernel='linear')
        svm_model.fit(embedding_scaled, ground_truth_labels)

        # Create grid to plot decision boundaries
        h = .02  # Step size in the mesh
        x_min, x_max = embedding_scaled[:, 0].min() - 1, embedding_scaled[:, 0].max() + 1
        y_min, y_max = embedding_scaled[:, 1].min() - 1, embedding_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Create a custom colormap from the label_to_color mapping
        unique_labels = np.unique(ground_truth_labels)
        unique_colors = [label_to_color[label] for label in unique_labels]
        cmap = mcolors.ListedColormap(unique_colors)

        # Plot decision boundary using the custom colormap
        Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Create the plot using scaled embeddings
        plt.scatter(embedding_scaled[:, 0], embedding_scaled[:, 1], c=[label_to_color[lbl] for lbl in ground_truth_labels], s=10, edgecolors='k', alpha=1)
        plt.contourf(xx, yy, Z, alpha=0.25, levels=np.linspace(Z.min(), Z.max(), 100), cmap=cmap, antialiased=True)

        plt.xlabel('UMAP 1st Component (scaled)')
        plt.ylabel('UMAP 2nd Component (scaled)')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.tight_layout()

    # Plot with color scheme "Time"
    if color_scheme == "Time":
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure and a 1x2 grid of subplots

        relative_labels = calculate_relative_position_labels(ground_truth_labels)
        relative_labels = np.array(relative_labels)
        plot_spectrogram_with_labels(spec_arr.T, relative_labels)

        axes[0].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=relative_labels, s=10, alpha=.1)
        axes[0].set_title("Time-based Coloring")

        # Plot with the original color scheme
        axes[1].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=[label_to_color[lbl] for lbl in ground_truth_labels], s=10, alpha=.1)
        axes[1].set_title("Original Coloring")

    else:
        # plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=[label_to_color[lbl] for lbl in ground_truth_labels], s=10, alpha=.1)
        # plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=ground_truth_labels, s=10, alpha=.1)  
        # plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=ground_truth_labels, s=10, alpha=.1)  



        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure and a 1x2 grid of subplots

        axes[0].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=hdbscan_labels, s=10, alpha=.1)
        axes[0].set_title("HDBSCAN")

        # Plot with the original color scheme
        axes[1].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=[label_to_color[lbl] for lbl in ground_truth_labels], s=10, alpha=.1)
        axes[1].set_title("Original Coloring")

    if raw_spectogram:
        plt.title(f'UMAP of Spectogram', fontsize=14)
    else:
        plt.title(f'UMAP Projection of (Layer: {layer_index}, Key: {dict_key})', fontsize=14)

    # Save the plot if save_dir is specified
    if save_dir:
        plt.savefig(save_dir, format='png')
    else:
        plt.show()

    # horrible code, not my fault... 
    if save_dict_for_analysis:
        # ## the remaning code has to do with creating a npz file dict that can be later used for analyzing this data 
        print(f"embedings arr {embedding_outputs.shape}")

        # start end 
        step_size = 2.7 * time_bins_per_umap_point

        emb_start = np.arange(0, step_size * embedding_outputs.shape[0], step_size)
        emb_end = emb_start + step_size 
        embStartEnd = np.stack((emb_start, emb_end), axis=0)
        print(f"embstartend {embStartEnd.shape}")

        colors_for_points = np.array([label_to_color[lbl] for lbl in ground_truth_labels])
        print(f"mean_colors_per_minispec {colors_for_points.shape}")

        colors_per_timepoint = []

        ground_truth_labels = ground_truth_labels.reshape(-1, 1)
        for label_row in ground_truth_labels:
            avg_color = label_to_color[int(label_row)]
            colors_per_timepoint.append(avg_color)
        colors_per_timepoint = np.array(colors_per_timepoint)

        print(f"colors_per_timepoint {colors_per_timepoint.shape}")

        embVals = embedding_outputs
        behavioralArr = spec_arr.T
        mean_colors_per_minispec = colors_for_points

        print(f"behavioral arr{behavioralArr.shape}")

        # Save the arrays into a single .npz file
        np.savez_compressed('/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/umap_dict_file.npz', 
                            embStartEnd=embStartEnd, 
                            embVals=embVals, 
                            behavioralArr=behavioralArr, 
                            mean_colors_per_minispec=mean_colors_per_minispec, 
                            colors_per_timepoint=colors_per_timepoint)
        
def similarity_of_vectors(model, device, data_dir="test_llb16",
                         remove_silences=False, samples=100, file_path='category_colors.pkl', 
                         layer_index=None, dict_key=None, time_bins_per_umap_point=100, 
                         context=1000, save_dir=None, raw_spectogram=False, save_dict_for_analysis=False, compute_svm=False, color_scheme="Syllable"):
    predictions_arr = []
    ground_truth_labels_arr = []
    spec_arr = [] 

    # Reset Figure
    plt.figure(figsize=(8, 6))

    # to allow sci notation 
    samples = int(samples)

    data_loader = load_data(data_dir=data_dir, remove_silences=remove_silences, context=context, psuedo_labels_generated=False)
    data_loader_iter = iter(data_loader)

    while len(ground_truth_labels_arr * context) < samples:
        # Because of the random windows being drawn from songs, it makes sense to reinit dataloader for UMAP plots 
        try:
            # Retrieve the next batch
            data, _, ground_truth_label = next(data_loader_iter)

        except StopIteration:
            # Reinitialize the DataLoader iterator when all batches are exhausted
            data_loader_iter = iter(data_loader)
            data, _, ground_truth_label = next(data_loader_iter)