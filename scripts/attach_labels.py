import numpy as np
import pandas as pd
import os

def label_spectrogram(npz_dir, annotation_csv, default_sample_rate=44100, NFFT=1024, step_size=119):
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
                onset_index = int(row['onset_Hz'] // step_size)  # Convert onset sample to spectrogram time frame
                offset_index = int(row['offset_Hz'] // step_size)  # Convert offset sample to spectrogram time frame
                label_value = int(row['label'])
                # Fill the label matrix with the label value within the event's time frame
                label_matrix[onset_index:offset_index] = label_value

            # Overwrite the old npz file with the new attribute
            np.savez(file_path, **data, labels=label_matrix)

            # For demonstration, just print out confirmation
            print(f"Labels added for {npz_file}: Label Matrix Shape {label_matrix.shape}, Spectrogram Shape {spectrogram_data.shape}")

# Example usage
npz_directory = "files/no_clip_test_llb3"
annotation_csv_path = '/media/george-vengrovski/disk2/canary_temp/llb3_data/llb3_annot.csv'
label_spectrogram(npz_directory, annotation_csv_path)