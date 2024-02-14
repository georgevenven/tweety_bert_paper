
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import re 
from collections import Counter
import umap
from data_class import SongDataSet_Image
from torch.utils.data import DataLoader
import glasbey

def syllable_to_phrase_labels(self, arr, silence=-1):
    new_arr = np.array(arr, dtype=int)
    current_syllable = None
    start_of_phrase_index = None
    first_non_silence_label = None  # To track the first non-silence syllable

    for i, value in enumerate(new_arr):
        if value != silence and value != current_syllable:
            if start_of_phrase_index is not None:
                new_arr[start_of_phrase_index:i] = current_syllable
            current_syllable = value
            start_of_phrase_index = i
            
            if first_non_silence_label is None:  # Found the first non-silence label
                first_non_silence_label = value

    if start_of_phrase_index is not None:
        new_arr[start_of_phrase_index:] = current_syllable

    # Replace the initial silence with the first non-silence syllable label
    if new_arr[0] == silence and first_non_silence_label is not None:
        for i in range(len(new_arr)):
            if new_arr[i] != silence:
                break
            new_arr[i] = first_non_silence_label

    return new_arr

def load_data( data_dir, context=1000, psuedo_labels_generated=True):
    dataset = SongDataSet_Image(data_dir, num_classes=196, remove_silences=False, psuedo_labels_generated=psuedo_labels_generated)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
    return loader 

def generate_hdbscan_labels(array, min_samples=5, min_cluster_size=3000):
    """
    Generate labels for data points using the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering algorithm.

    Parameters:
    - array: ndarray of shape (n_samples, n_features)
      The input data to cluster.

    - min_samples: int, default=5
      The number of samples in a neighborhood for a point to be considered as a core point.

    - min_cluster_size: int, default=5
      The minimum number of points required to form a cluster.

    Returns:
    - labels: ndarray of shape (n_samples)
      Cluster labels for each point in the dataset. Noisy samples are given the label -1.
    """

    import hdbscan

    # Create an HDBSCAN object with the specified parameters.
    hdbscan_model = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)

    # Fit the model to the data and extract the labels.
    labels = hdbscan_model.fit_predict(array)

    print(np.unique(labels))

    return labels

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
    total_samples = 0

    data_loader = load_data(data_dir=data_dir, context=context, psuedo_labels_generated=True)
    data_loader_iter = iter(data_loader)

    while len(ground_truth_labels_arr) * context < samples:
        try:
            # Retrieve the next batch
            data, _, ground_truth_label = next(data_loader_iter)

            # if smaller than context window, go to next song
            if data.shape[1] < context:
                continue 
            # because network is made to work with batched data, we unsqueeze a dim and transpose the last two dims (usually handled by collate fn)
            data = data.unsqueeze(0).permute(0,1,3,2)

            # calculate the number of times a song 
            num_times = data.shape[-1] // context
            
            # removing left over timebins that do not fit in context window 
            shave_index = num_times * context
            data = data[:,:,:,:shave_index]

            batch, channel, freq, time_bins = data.shape 

            # cheeky reshaping operation to reshape the length of the song that is larger than the context window into multiple batches 
            data = data.permute(0,-1, 1, 2)
            data = data.reshape(num_times, context, channel, freq)
            data = data.permute(0,2,3,1)

            # reshaping g truth labels to be consistent 
            batch, time_bins, labels = ground_truth_label.shape

            # shave g truth labels 
            ground_truth_label = ground_truth_label.permute(0,2,1)
            ground_truth_label = ground_truth_label[:,:,:shave_index]

            # cheeky reshaping operation to reshape the length of the song that is larger than the context window into multiple batches 
            ground_truth_label = ground_truth_label.permute(0,2,1)
            ground_truth_label = ground_truth_label.reshape(num_times, context, labels)
            
        except StopIteration:
            # if test set is exhausted, print the number of samples collected and stop the collection process
            print(f"samples collected f{len(ground_truth_labels_arr) * context}")

        if raw_spectogram == False:
            _, layers = model.inference_forward(data.to(device))

            layer_output_dict = layers[layer_index]
            output = layer_output_dict.get(dict_key, None)

            if output is None:
                print(f"Invalid key: {dict_key}. Skipping this batch.")
                continue

            batches, time_bins, features = output.shape 
            # data shape [0] is the number of batches, 
            predictions = output.reshape(batches, time_bins, features)
            # combine the batches and number of samples per context window 
            predictions = predictions.flatten(0,1)
            predictions_arr.append(predictions.detach().cpu().numpy())

        # remove channel dimension 
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
        
        total_samples += spec.shape[0]

    # convert the list of batch * samples * features to samples * features 
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
    hdbscan_labels = generate_hdbscan_labels(embedding_outputs)
    ground_truth_labels = syllable_to_phrase_labels(ground_truth_labels)
    np.savez("files/labels.npy", embedding_outputs=embedding_outputs, hdbscan_labels=hdbscan_labels, ground_truth_labels=ground_truth_labels)

    # np.savez_compressed('hdbscan_and_gtruth.npz', ground_truth_labels=ground_truth_labels, embedding_outputs=embedding_outputs, hdbscan_labels=hdbscan_labels)

    cmap = glasbey.extend_palette(["#000000"], palette_size=30)
    cmap = mcolors.ListedColormap(cmap)    

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure and a 1x2 grid of subplots

    axes[0].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=hdbscan_labels, s=10, alpha=.1, cmap=cmap)
    axes[0].set_title("HDBSCAN")

    # Plot with the original color scheme
    axes[1].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=ground_truth_labels, s=10, alpha=.1, cmap=cmap)
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


    # # horrible code, not my fault... 
    # if save_dict_for_analysis:
    #     # ## the remaning code has to do with creating a npz file dict that can be later used for analyzing this data 
    #     print(f"embedings arr {embedding_outputs.shape}")

    #     # start end 
    #     step_size = 2.7 * time_bins_per_umap_point

    #     emb_start = np.arange(0, step_size * embedding_outputs.shape[0], step_size)
    #     emb_end = emb_start + step_size 
    #     embStartEnd = np.stack((emb_start, emb_end), axis=0)
    #     print(f"embstartend {embStartEnd.shape}")

    #     colors_for_points = np.array([label_to_color[lbl] for lbl in ground_truth_labels])
    #     print(f"mean_colors_per_minispec {colors_for_points.shape}")

    #     colors_per_timepoint = []

    #     ground_truth_labels = ground_truth_labels.reshape(-1, 1)
    #     for label_row in ground_truth_labels:
    #         avg_color = label_to_color[int(label_row)]
    #         colors_per_timepoint.append(avg_color)
    #     colors_per_timepoint = np.array(colors_per_timepoint)

    #     print(f"colors_per_timepoint {colors_per_timepoint.shape}")

    #     embVals = embedding_outputs
    #     behavioralArr = spec_arr.T
    #     mean_colors_per_minispec = colors_for_points

    #     print(f"behavioral arr{behavioralArr.shape}")

    #     # Save the arrays into a single .npz file
    #     np.savez_compressed('/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/umap_dict_file.npz', 
    #                         embStartEnd=embStartEnd, 
    #                         embVals=embVals, 
    #                         behavioralArr=behavioralArr, 
    #                         mean_colors_per_minispec=mean_colors_per_minispec, 
    #                         colors_per_timepoint=colors_per_timepoint)

class AnalyzeSyntax():
    def __init__(self):
            pass

    def syllable_to_phrase_labels(self, arr, silence=-1):
        new_arr = np.array(arr, dtype=int)
        current_syllable = None
        start_of_phrase_index = None
        first_non_silence_label = None  # To track the first non-silence syllable

        for i, value in enumerate(new_arr):
            if value != silence and value != current_syllable:
                if start_of_phrase_index is not None:
                    new_arr[start_of_phrase_index:i] = current_syllable
                current_syllable = value
                start_of_phrase_index = i
                
                if first_non_silence_label is None:  # Found the first non-silence label
                    first_non_silence_label = value

        if start_of_phrase_index is not None:
            new_arr[start_of_phrase_index:] = current_syllable

        # Replace the initial silence with the first non-silence syllable label
        if new_arr[0] == silence and first_non_silence_label is not None:
            for i in range(len(new_arr)):
                if new_arr[i] != silence:
                    break
                new_arr[i] = first_non_silence_label

        return new_arr

    def reduce_phrases(self, arr, remove_silence=True):
        current_element = arr[0]
        reduced_list = [] 

        for i, value in enumerate(arr):
            if value != current_element:
                reduced_list.append(current_element)
                current_element = value 

            # append last phrase
            if i == len(arr) - 1:
                reduced_list.append(current_element)

        if remove_silence == True:
            reduced_list = [value for value in reduced_list if value != 0]

        return np.array(reduced_list)

    def majority_vote(self, data):
        # Function to find the majority element in a window
        def find_majority(window):
            count = Counter(window)
            majority = max(count.values())
            for num, freq in count.items():
                if freq == majority:
                    return num
            return window[1]  # Return the middle element if no majority found

        # Ensure the input data is in list form
        if isinstance(data, str):
            data = [int(x) for x in data.split(',') if x.strip().isdigit()]

        # Initialize the output array with a padding at the beginning
        output = [data[0]]  # Pad with the first element

        # Apply the majority vote on each window
        for i in range(1, len(data) - 1):  # Start from 1 and end at len(data) - 1 to avoid index out of range
            window = data[i-1:i+2]  # Define the window with size 3
            output.append(find_majority(window))

        # Pad the output array at the end to match the input array size
        output.append(data[-1])

        return output

    def integer_to_letter(self, match):
        """Converts an integer match to a corresponding letter (1 -> A, 2 -> B, etc.)."""
        num = int(match.group())
        # Subtract 1 from the number to get 0-based indexing for letters, then mod by 26 to handle numbers > 26
        return chr((num - 1) % 26 + ord('A'))

    def replace_integers_with_letters(self, file_path):
        """Reads a file, replaces all integers with their corresponding letters, and writes the changes back to the file."""
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Replace all occurrences of integers in the file with their corresponding letters
        modified_content = re.sub(r'\b\d+\b', self.integer_to_letter, content)
        
        with open(file_path, 'w') as file:
            file.write(modified_content)    

    def syllable_to_phrase_labels(self, arr, silence=0):
        new_arr = np.array(arr, dtype=int)
        current_syllable = None
        start_of_phrase_index = None
        first_non_silence_label = None  # To track the first non-silence syllable

        for i, value in enumerate(new_arr):
            if value != silence and value != current_syllable:
                if start_of_phrase_index is not None:
                    new_arr[start_of_phrase_index:i] = current_syllable
                current_syllable = value
                start_of_phrase_index = i
                
                if first_non_silence_label is None:  # Found the first non-silence label
                    first_non_silence_label = value

        if start_of_phrase_index is not None:
            new_arr[start_of_phrase_index:] = current_syllable

        # Replace the initial silence with the first non-silence syllable label
        if new_arr[0] == silence and first_non_silence_label is not None:
            for i in range(len(new_arr)):
                if new_arr[i] != silence:
                    break
                new_arr[i] = first_non_silence_label

        return new_arr