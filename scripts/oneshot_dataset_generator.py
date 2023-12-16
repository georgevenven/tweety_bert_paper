import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.append("src")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

# purpose of file: go through eval folder, remove silences (0 label), collect statistics about the remaining classes (median length)
# Afterwards, generate an npz spectogram which is a combination of each phrase type appended with silence at the end as well (0 label)

eval_dir = "/home/george-vengrovski/Documents/data/eval_dataset/llb3_data_matrices"
save_dir = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files"

class_segments = {}
segment_indices = {}
segment_files = {}

files = [f for f in os.listdir(eval_dir) if f.endswith(".npz")]

# Collect Statistics 
for file in tqdm(files, desc="Processing files"):
    file_path = os.path.join(eval_dir, file)
    

# Create and save the combined spectrogram
final_spectrogram = None
for key, segment in median_segmentfile_path = os.path.join(eval_dir, file)_index:end_index]

    if final_spectrogram is None:
        final_spectrogram = snippet
    else:
        final_spectrogram = np.hstack((final_spectrogram, snippet))

# Save the final .npz file
final_path = os.path.join(save_dir, "combined_spectrogram.npz")
np.savez(final_path, s=final_spectrogram)

print("Combined spectrogram saved.")



        # new_labels = f['labels']

        # # Z-score normalization
        # mean_val = spectogram.mean()
        # std_val = spectogram.std()
        # spectogram = (spectogram - mean_val) / (std_val + 1e-7)  # Adding a small constant to prevent division by zero
        
        # # Replace NaN values with zeros
        # spectogram[np.isnan(spectogram)] = 0


        # # Crop the spectrogram (assuming this is intended)
        # f_dict = {'s': spectogram[20:216,:], 'labels': labels, 'new_labels': np.zeros(new_labels.shape)}

        
        # # Process the data
        # # Remove silences, collect statistics, etc.

        # # Generate combined spectrogram
        # # Append silence at the end

        # # Save the final spectrogram
        # save_path = os.path.join(save_dir, "processed_" + file)
        # np.savez(save_path, spectrogram=combined_spectrogram)