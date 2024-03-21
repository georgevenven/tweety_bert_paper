import numpy as np
import pandas as pd
import os

def label_spectrogram(npz_dir, annotation_csv, default_sample_rate=44100, NFFT=1024, step_size=512):
    # Load the annotations CSV
    annotations = pd.read_csv(annotation_csv)

    # Iterate over npz files
    for npz_file in os.listdir(npz_dir):
        if npz_file.endswith('.npz'):
            file_path = os.path.join(npz_dir, npz_file)
            data = np.load(file_path, allow_pickle=True)
            spectrogram_data = data['s']  # Replace 's' with the actual key for the spectrogram

            # Extract the basename without the extension for matching
            base_filename = os.path.splitext(npz_file)[0]

            # Get corresponding rows from the annotations DataFrame
            relevant_annotations = annotations[annotations['audio_file'].str.contains(base_filename)]

            # Initialize the label matrix as a 1D array with the same number of time frames as the spectrogram
            label_matrix = np.zeros(spectrogram_data.shape[1], dtype=int)

            # Loop over each annotation and fill in the label matrix
            for _, row in relevant_annotations.iterrows():
                onset_seconds = row['onset_Hz'] / default_sample_rate
                offset_seconds = row['offset_Hz'] / default_sample_rate
                onset_index = int(onset_seconds * default_sample_rate // step_size)
                offset_index = int(offset_seconds * default_sample_rate // step_size)
                label_value = int(row['label'])
                
                # Fill the label matrix with the label value within the event's time frame
                label_matrix[onset_index:offset_index] = label_value

            # Overwrite the old npz file with the new attribute
            print(label_matrix)
            np.savez(file_path, s=spectrogram_data, labels=label_matrix)

            # For demonstration, just print out confirmation
            print(f"Labels added for {npz_file}: Label Matrix Shape {label_matrix.shape}, Spectrogram Shape {spectrogram_data.shape}")


# Example usage
npz_directory = "/media/george-vengrovski/disk1/combined_llb3_test"
annotation_csv_path = '/media/george-vengrovski/disk2/canary_yarden/llb3_data/llb3_annot.csv'
label_spectrogram(npz_directory, annotation_csv_path)