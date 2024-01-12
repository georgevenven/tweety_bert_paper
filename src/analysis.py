import os
import pickle
import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_class import SongDataSet_Image, CollateFunction
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


def load_data( data_dir,remove_silences=False, context=1000, psuedo_labels_generated=False):
    collate_fn = CollateFunction(context)
    dataset = SongDataSet_Image(data_dir, num_classes=196, remove_silences=remove_silences, psuedo_labels_generated=psuedo_labels_generated)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=16)
    return loader 

def plot_umap_projection(model, device, data_dir="test_llb16",
                         remove_silences=False, samples=100, file_path='category_colors.pkl', 
                         layer_index=None, dict_key=None, time_bins_per_umap_point=100, 
                         context=1000, save_dir=None, raw_spectogram=False, save_dict_for_analysis=False):
    predictions_arr = []
    ground_truth_labels_arr = []
    spec_arr = [] 

    data_loader = load_data(data_dir=data_dir, remove_silences=remove_silences, context=context, psuedo_labels_generated=False)
    
    for i, (data, _, ground_truth_label) in enumerate(data_loader):
        if i == samples:
            break

        spec = data 

        if raw_spectogram == False:
            output, layers = model.inference_forward(data.to(device))
            layer_output_dict = layers[layer_index]
            temp = layer_output_dict.get(dict_key, None)

            if temp is None:
                print(f"Invalid key: {dict_key}. Skipping this batch.")
                continue

            data = temp

        num_times = ground_truth_label.shape[1] / time_bins_per_umap_point

        predictions = data.reshape(data.shape[0] * int(num_times), time_bins_per_umap_point, -1)
        predictions = predictions.flatten(1, 2)
        
        ground_truth_label = ground_truth_label.reshape(data.shape[0] * int(num_times), time_bins_per_umap_point, ground_truth_label.shape[2])
        ground_truth_label = torch.argmax(ground_truth_label, dim=-1)
        ground_truth_label = ground_truth_label.squeeze(1)

        spec = spec.reshape(spec.shape[2], -1)
        spec = spec.T 
        
        spec_arr.append(spec.detach().cpu().numpy())
        predictions_arr.append(predictions.detach().cpu().numpy())
        ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
        
    with open(file_path, 'rb') as file:
        color_map_data = pickle.load(file)

    label_to_color = {label: tuple(color) for label, color in color_map_data.items()}
    
    predictions = np.concatenate(predictions_arr, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
    spec_arr = np.concatenate(spec_arr, axis=0)

    colors_for_points = []
    for label_row in ground_truth_labels:
        if label_row.ndim > 0:
            # If label_row is iterable (more than one dimension)
            row_colors = [label_to_color[int(lbl)] for lbl in label_row]
            avg_color = np.mean(row_colors, axis=0)
        else:
            # If label_row is a single integer (one dimension)
            avg_color = label_to_color[int(label_row)]
        colors_for_points.append(avg_color)
    
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.05, n_components=2, metric='euclidean')
    embedding_outputs = reducer.fit_transform(predictions)
    
    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], s=10, c=colors_for_points, alpha=.5)

    if raw_spectogram == True:
        plt.title(f'UMAP of Spectogram', fontsize=14)
    else:
        plt.title(f'UMAP projection of the model (Layer: {layer_index}, Key: {dict_key})', fontsize=14)
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



 

def plot_pca_projection(model, device, data_dir="test_llb16", 
                        remove_silences=False, samples=100, file_path='category_colors.pkl', 
                        layer_index=None, dict_key=None, time_bins_per_pca_point=100, 
                        context=1000, save_dir=None, raw_spectogram=False):
    predictions_arr = []
    ground_truth_labels_arr = []

    # Assuming load_data is a function you have defined elsewhere
    data_loader = load_data(data_dir=data_dir,
                            remove_silences=remove_silences, context=context, 
                            psuedo_labels_generated=False)
    
    for i, (data, _, ground_truth_label) in enumerate(data_loader):
        if i == samples:
            break

        if not raw_spectogram:
            output, layers = model.inference_forward(data.to(device))
            layer_output_dict = layers[layer_index]
            temp = layer_output_dict.get(dict_key, None)

            if temp is None:
                print(f"Invalid key: {dict_key}. Skipping this batch.")
                continue
            
            data = temp 

        num_times = ground_truth_label.shape[1] / time_bins_per_pca_point

        predictions = data.reshape(data.shape[0] * int(num_times), time_bins_per_pca_point, -1)
        predictions = predictions.flatten(1, 2)
        
        ground_truth_label = ground_truth_label.reshape(data.shape[0] * int(num_times), time_bins_per_pca_point, ground_truth_label.shape[2])
        ground_truth_label = torch.argmax(ground_truth_label, dim=-1)
        ground_truth_label = ground_truth_label.squeeze(1)
        
        predictions_arr.append(predictions.detach().cpu().numpy())
        ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
        
    with open(file_path, 'rb') as file:
        color_map_data = pickle.load(file)

    label_to_color = {label: tuple(color) for label, color in color_map_data.items()}
    
    predictions = np.concatenate(predictions_arr, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)

    colors_for_points = []
    for label_row in ground_truth_labels:
        if label_row.ndim > 0:
            row_colors = [label_to_color[int(lbl)] for lbl in label_row]
            avg_color = np.mean(row_colors, axis=0)
        else:
            avg_color = label_to_color[int(label_row)]
        colors_for_points.append(avg_color)
    
    pca = PCA(n_components=2, random_state=42)
    embedding_outputs = pca.fit_transform(predictions)
    
    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], s=10, c=colors_for_points, alpha=.5)

    if raw_spectogram:
        plt.title(f'PCA of Raw Spectogram', fontsize=14)
    else:
        plt.title(f'PCA projection of the model (Layer: {layer_index}, Key: {dict_key})', fontsize=14)
    plt.tight_layout()

    # Save the plot if save_dir is specified
    if save_dir:
        plt.savefig(save_dir, format='png')
     
    plt.show()