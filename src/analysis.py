import pickle
import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_class import SongDataSet_Image, CollateFunction
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance
import hdbscan
import os 
from hmmlearn import hmm

def load_data( data_dir, remove_silences=False, context=1000, psuedo_labels_generated=True):
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

def generate_hdbscan_labels(array, min_cluster_size=500, min_samples=1):
    """
    Applies HDBSCAN clustering to an array of data points and returns the cluster labels.

    Args:
        array (numpy.ndarray): An n x features array of data points.
        min_cluster_size (int): The minimum size of clusters; smaller clusters will be considered as noise.

    Returns:
        numpy.ndarray: An n x 1 array of cluster labels for each data point, where -1 indicates noise points.
    """

    # Initialize the HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', min_samples=min_samples)

    # Fit the model to the data
    clusterer.fit(array)

    # Extract the cluster labels
    labels = clusterer.labels_

    # Optionally, reshape labels to n x 1 if you specifically need this shape
    labels = labels.reshape(-1, 1)

    # Print the unique labels to see the different clusters and noise
    print(np.unique(labels))

    return labels

def generate_kmeans_labels(array, n_clusters=10):
    """
    Applies K-means clustering to an array of data points and returns the cluster labels.

    Args:
        array (numpy.ndarray): An n x features array of data points.
        n_clusters (int): The number of clusters to form.

    Returns:
        numpy.ndarray: An n x 1 array of cluster labels for each data point.
    """

    # Initialize the KMeans clusterer
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit the model to the data and predict the cluster labels
    labels = kmeans.fit_predict(array)

    # Optionally, reshape labels to n x 1 if you specifically need this shape
    labels = labels.reshape(-1, 1)

    # Print the unique labels to see the different clusters
    print(np.unique(labels))

    return labels


def generate_hmm_states(array, n_states=30, context=1000):
    """
    Applies Hidden Markov Model to an array of data points and returns the state labels.

    Args:
        array (numpy.ndarray): An n x features array of data points, representing a sequence of observations.
        n_states (int): The number of hidden states in the model.

    Returns:
        numpy.ndarray: An n x 1 array of state labels for each observation.
    """
    
    # Initialize the HMM
    # We'll use a Gaussian HMM here, but you might choose a different type based on your data
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=0, verbose=True)

    # Fit the model to the data
    # HMMs require the lengths of the sequences if the dataset contains multiple sequences
    # We calculate that here 
    lengths = [context] * (len(array) // context)


    model.fit(array, lengths=lengths)

    # Predict the hidden states for each observation in the array
    states = model.predict(array)

    # Optionally, reshape states to n x 1 if you specifically need this shape
    states = states.reshape(-1, 1)

    # Print the unique states to see the different inferred hidden states
    print(np.unique(states))

    return states

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

    data_loader = load_data(data_dir=data_dir, remove_silences=remove_silences, context=context, psuedo_labels_generated=True)
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
    reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='cosine')

    embedding_outputs = reducer.fit_transform(predictions)
    # hdbscan_labels = generate_hdbscan_labels(embedding_outputs)
    hdbscan_labels = generate_hmm_states(embedding_outputs, context=context)

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


import numpy as np
from scipy.spatial import distance
import random
import matplotlib.pyplot as plt

def average_similarity_between_samples(group1, group2, sample_size=30):
    """
    Calculates the average cosine similarity and Euclidean distance between samples of two groups of vectors.
    """
    # Sample vectors from both groups
    group1_sample = random.sample(group1, min(len(group1), sample_size))
    group2_sample = random.sample(group2, min(len(group2), sample_size))

    # Convert to numpy arrays for distance calculation
    group1_sample = np.stack(group1_sample)
    group2_sample = np.stack(group2_sample)

    # Calculate pairwise cosine and Euclidean distances
    cosine_distances = distance.cdist(group1_sample, group2_sample, 'cosine')
    euclidean_distances = distance.cdist(group1_sample, group2_sample, 'euclidean')

    # Convert cosine distances to similarities
    cosine_similarities = 1 - cosine_distances

    # Calculate average similarities/distances
    return np.mean(cosine_similarities), np.mean(euclidean_distances)

def compare_across_keys(vectors_dict, sample_size=30):
    """
    Compares vectors across keys by calculating average similarities between sampled subsets, including self comparisons.
    """
    keys = list(vectors_dict.keys())
    comparisons = {}

    for key1 in keys:
        for key2 in keys:  # Compare with itself and every other key
            avg_cosine_sim, avg_euclidean_dist = average_similarity_between_samples(vectors_dict[key1], vectors_dict[key2], sample_size)
            comparisons[(key1, key2)] = {'Average Cosine Similarity': avg_cosine_sim, 'Average Euclidean Distance': avg_euclidean_dist}

    return comparisons

def create_similarity_matrices(comparisons, keys):
    """
    Creates matrices for cosine similarities and Euclidean distances based on comparisons.
    """
    n = len(keys)
    cosine_matrix = np.zeros((n, n))
    euclidean_matrix = np.zeros((n, n))
    key_to_index = {key: i for i, key in enumerate(keys)}

    for (key1, key2), metrics in comparisons.items():
        i, j = key_to_index[key1], key_to_index[key2]
        cosine_matrix[i, j] = metrics['Average Cosine Similarity']
        euclidean_matrix[i, j] = metrics['Average Euclidean Distance']

    return cosine_matrix, euclidean_matrix

def plot_similarity_matrix(matrix, title, cmap='viridis', save_dir=None):
    """
    Plots a similarity matrix and saves the plot to a file if save_dir is provided.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Key Index')
    plt.ylabel('Key Index')

    if save_dir is not None:
        # Ensure the save_dir exists, create if not
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{title.replace(' ', '_')}.png"  # Replace spaces with underscores for the filename
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()  # Close the plot to free up memory
        print(f"Plot saved as {filepath}")
    else:
        plt.show()

def similarity_of_vectors(model, device, data_dir="test_llb16",
                         remove_silences=False, samples=100, file_path='category_colors.pkl', 
                         layer_index=1, dict_key="V",
                         context=1000, save_dir=None, raw_spectogram=False):

    raw_spec_vectors = {}
    neural_activation_vectors = {}

    # Reset Figure
    plt.figure(figsize=(8, 6))

    # to allow sci notation 
    samples = int(samples)

    data_loader = load_data(data_dir=data_dir, remove_silences=remove_silences, context=context, psuedo_labels_generated=True)
    data_loader_iter = iter(data_loader)

    i = 0
    while i < samples:
        # Because of the random windows being drawn from songs, it makes sense to reinit dataloader for UMAP plots 
        try:
            # Retrieve the next batch
            data, _, ground_truth_label = next(data_loader_iter)

        except StopIteration:
            # Reinitialize the DataLoader iterator when all batches are exhausted
            data_loader_iter = iter(data_loader)
            data, _, ground_truth_label = next(data_loader_iter)

        _, layers = model.inference_forward(data.to(device))

        layer_output_dict = layers[layer_index]
        output = layer_output_dict.get(dict_key, None)

        if output is None:
            print(f"Invalid key: {dict_key}. Skipping this batch.")
            continue

        ground_truth_label = ground_truth_label.argmax(-1).flatten()

        # reshape to remove batch channel, and make time sequence first dim
        data = data[0,0].T

        # remove batch and make time sequence the first dimension 
        output = output[0]

        for idx, label in enumerate(ground_truth_label):
            label_item = label.item()

            if label_item not in raw_spec_vectors:
                raw_spec_vectors[label_item] = []
            raw_spec_vectors[label_item].append(output[idx].detach().cpu().numpy())

            if label_item not in neural_activation_vectors:
                neural_activation_vectors[label_item] = []
            neural_activation_vectors[label_item].append(data[idx].detach().cpu().numpy())

        i += 1  # Increment the loop counter

    SAMPLE_SIZE = 100  # Adjust based on your dataset size and computational resources

    # Assuming raw_spec_vectors and neural_activation_vectors are already populated
    raw_spec_comparisons = compare_across_keys(raw_spec_vectors, sample_size=SAMPLE_SIZE)
    neural_activation_comparisons = compare_across_keys(neural_activation_vectors, sample_size=SAMPLE_SIZE)

    # Create and plot similarity matrices for raw spectrogram vectors
    raw_spec_keys = list(raw_spec_vectors.keys())
    raw_cosine_matrix, raw_euclidean_matrix = create_similarity_matrices(raw_spec_comparisons, raw_spec_keys)
    plot_similarity_matrix(raw_cosine_matrix, 'Average Cosine Similarity Matrix for Raw Spectrograms', save_dir='results/vec_comp')
    # plot_similarity_matrix(raw_euclidean_matrix, 'Average Euclidean Distance Matrix for Raw Spectrograms', cmap='magma')

    # Create and plot similarity matrices for neural activation vectors
    neural_activation_keys = list(neural_activation_vectors.keys())
    neural_cosine_matrix, neural_euclidean_matrix = create_similarity_matrices(neural_activation_comparisons, neural_activation_keys)
    plot_similarity_matrix(neural_cosine_matrix, 'Average Cosine Similarity Matrix for Neural Activations', save_dir='results/vec_comp')
    # plot_similarity_matrix(neural_euclidean_matrix, 'Average Euclidean Distance Matrix for Neural Activations', cmap='magma')

    # Print comparison metrics for raw spectrogram vectors
    for key_pair, metrics in raw_spec_comparisons.items():
        print(f"Raw Spec Vectors - Key Pair: {key_pair}, Metrics: {metrics}")

    # Print comparison metrics for neural activation vectors
    for key_pair, metrics in neural_activation_comparisons.items():
        print(f"Neural Activation Vectors - Key Pair: {key_pair}, Metrics: {metrics}")