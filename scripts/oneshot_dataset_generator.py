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
    
    # Load the data
    data = np.load(file_path)
    spectrogram = data['s']
    labels = data['labels'].flatten()

    # Vectorized removal of silences and preserving original indices
    original_indices = np.arange(len(labels))
    non_silence_indices = labels != 0
    labels_no_silence = labels[non_silence_indices]
    original_indices_no_silence = original_indices[non_silence_indices]

    # Segment processing with preserved indices
    prev_label = -1
    start_index_original = 0
    for i, label in enumerate(labels_no_silence):
        if label != prev_label:
            if prev_label != -1:
                end_index_original = original_indices_no_silence[i - 1] + 1
                seg_key = (prev_label, end_index_original - start_index_original)

                if seg_key not in class_segments:
                    class_segments[seg_key] = []
                    segment_indices[seg_key] = []
                    segment_files[seg_key] = []

                class_segments[seg_key].append(labels[start_index_original:end_index_original])
                segment_indices[seg_key].append((start_index_original, end_index_original))
                segment_files[seg_key].append(file)

            start_index_original = original_indices_no_silence[i]
            prev_label = label

    # Handle the last segment if not silence
    if prev_label != 0:
        end_index_original = original_indices_no_silence[-1] + 1
        seg_key = (prev_label, end_index_original - start_index_original)
        if seg_key not in class_segments:
            class_segments[seg_key] = []
            segment_indices[seg_key] = []
            segment_files[seg_key] = []
        class_segments[seg_key].append(labels[start_index_original:end_index_original])
        segment_indices[seg_key].append((start_index_original, end_index_original))
        segment_files[seg_key].append(file)

# Calculate median lengths and select snippets
median_segments = {}
for key, segments in class_segments.items():
    lengths = [len(seg) for seg in segments]
    median_length = np.median(lengths)
    closest_segment = min(segments, key=lambda x: abs(len(x) - median_length))
    median_segments[key] = closest_segment

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