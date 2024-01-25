import os
import pickle
import umap
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import torch
from data_class import SongDataSet_Image, CollateFunction
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors


def load_data( data_dir,remove_silences=False, context=1000, psuedo_labels_generated=False):
    collate_fn = CollateFunction(context)
    dataset = SongDataSet_Image(data_dir, num_classes=196, remove_silences=False, psuedo_labels_generated=psuedo_labels_generated)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=16)
    return loader 

def calculate_relative_position_labels(ground_truth_labels, silence=0):
    """
    Creates a new list, which labels each timebin's relative position within each phrase.

    Args:
        ground_truth_labels (List[int]): The list of integer values representing timebin syllable labels.
        silence (int): The label representing silence, used to separate phrases.

    Returns:
        relative_position_labels (List[float]): List of float values between 0-1 representing the position of the timebin within each phrase.
    """

    # Convert list to numpy array for vectorized operations
    labels_array = np.array(ground_truth_labels)

    # Split the labels into phrases separated by silence
    # Find indices where syllables are not silence
    non_silence_indices = np.where(labels_array != silence)[0]

    # Initialize an array to store relative positions
    relative_positions = np.zeros_like(labels_array, dtype=float)

    # Find start and end indices of each phrase
    if len(non_silence_indices) > 0:
        phrase_starts = np.concatenate(([non_silence_indices[0]], non_silence_indices[:-1][np.diff(non_silence_indices) > 1]))
        phrase_ends = np.concatenate((non_silence_indices[:-1][np.diff(non_silence_indices) > 1], [non_silence_indices[-1]]))

        # Process each phrase
        for start, end in zip(phrase_starts, phrase_ends):
            phrase_length = end - start + 1
            relative_positions[start:end+1] = np.arange(1, phrase_length + 1) / phrase_length

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

    while len(predictions_arr) < samples:

        # Because of the random windows being drawn from songs, it makes sense to reinit dataloader for UMAP plots 
        try:
            # Retrieve the next batch
            data, _, ground_truth_label = next(data_loader_iter)

        except StopIteration:
            # Reinitialize the DataLoader iterator when all batches are exhausted
            data_loader_iter = iter(data_loader)
            data, _, ground_truth_label = next(data_loader_iter)

        if raw_spectogram == False:
            embedding_output, layers, feature_extractor = model.inference_forward(data.to(device))

            if dict_key == "conv":
                print(feature_extractor.shape)
                output = feature_extractor.permute(0,2,1)
            elif dict_key == "embedding_output":
                pass
            else:
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
            
            # reshape the labels just like the neural activations (last one is different size from features and is of size of one hot encoding)
            ground_truth_label = ground_truth_label.reshape(batches, num_times, time_bins_per_umap_point, -1)
            ground_truth_label = ground_truth_label.flatten(0,1)
            ground_truth_label = torch.argmax(ground_truth_label, dim=-1)

        else:
            # number of times the spectogram must be broken apart 
            num_times = context // time_bins_per_umap_point

            data = data.squeeze(1)
           
            batches, features, timebins = data.shape 

            # data shape [0] is the number of batches, 
            predictions = data.reshape(batches, features, num_times, time_bins_per_umap_point)
   
            # combine the batches and number of samples per context window 
            predictions = predictions.flatten(0,1)
            
            # combine features and time bins
            predictions = predictions.flatten(-2,-1)

            # reshape the labels just like the neural activations (last one is different size from features and is of size of one hot encoding)
            ground_truth_label = ground_truth_label.reshape(batches, num_times, time_bins_per_umap_point, -1)
            ground_truth_label = ground_truth_label.flatten(0,1)
            ground_truth_label = torch.argmax(ground_truth_label, dim=-1)
            
        predictions_arr.append(predictions.detach().cpu().numpy())
        ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
        
    with open(file_path, 'rb') as file:
        color_map_data = pickle.load(file)

    label_to_color = {label: tuple(color) for label, color in color_map_data.items()}
    
    predictions = np.concatenate(predictions_arr, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
    
    # dims is 1 if no windowing is utilized
    _, dims = ground_truth_labels.shape
    
    if dims != 1:
        ground_truth_labels, _ = mode(ground_truth_labels, axis=1)
    ground_truth_labels = ground_truth_labels.flatten()

    # razor off any extra datapoints 
    if samples > len(predictions):
        samples = len(predictions)
    else:
        predictions = predictions[:samples]
        ground_truth_labels = ground_truth_labels[:samples]
        
    # Fit the UMAP reducer
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.05, n_components=2, metric='cosine')
    embedding_outputs = reducer.fit_transform(predictions)

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

        # Create the plot
        contour = plt.contourf(xx, yy, Z, alpha=0.25, levels=np.linspace(Z.min(), Z.max(), 100), cmap=cmap, antialiased=True)
        # possibly replace c= with lamda function 
        plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=[label_to_color[lbl] for lbl in ground_truth_labels], s=10, edgecolors='k', alpha=1)


    # # Plot with color scheme "Time"
    # if color_scheme == "Time":
    #     fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure and a 1x2 grid of subplots

    #     relative_labels = calculate_relative_position_labels(ground_truth_labels)
    #     axes[0].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=relative_labels, s=10, alpha=.1)
    #     axes[0].set_title("Time-based Coloring")

    #     # Plot with the original color scheme
    #     axes[1].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=[label_to_color[lbl] for lbl in ground_truth_labels], s=10, alpha=.1)
    #     axes[1].set_title("Original Coloring")

    # else:
    plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=[label_to_color[lbl] for lbl in ground_truth_labels], s=10, alpha=.1) 

    if raw_spectogram:
        plt.title(f'UMAP of Spectogram', fontsize=14)
    else:
        plt.title(f'UMAP Projection of (Layer: {layer_index}, Key: {dict_key})', fontsize=14)

    plt.xlabel('UMAP 1st Component')
    plt.ylabel('UMAP 2nd Component')
    
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    plt.tight_layout()

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

        colors_for_points = np.array(colors_for_points)
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