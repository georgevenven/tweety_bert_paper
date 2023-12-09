import numpy as np
import matplotlib.pyplot as plt
import os
import shutil 
import random 
import sys

sys.path.append("src")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_cluster/project')

def copy_folder_with_removal(src, dst, removal_percentage=50):
    """
    Copies only .npz files from `src` to `dst`, removing a percentage of these files in the process.

    :param src: Source directory path.
    :param dst: Destination directory path.
    :param removal_percentage: Percentage of .npz files to be removed during copying. Default is 50.
    """
    if not os.path.exists(src):
        raise ValueError(f"Source directory {src} does not exist.")
    
    if os.path.exists(dst):
        raise ValueError(f"Destination directory {dst} already exists.")

    # Create destination directory
    os.makedirs(dst)

    # List all npz files in the source directory
    npz_files = [item for item in os.listdir(src) if item.endswith('.npz')]

    # Calculate number of npz files to remove
    files_to_remove = int(len(npz_files) * removal_percentage / 100)

    # Randomly select npz files to remove
    npz_files_removed = set(random.sample(npz_files, files_to_remove))

    # Copy npz files, skipping the selected ones
    for file in npz_files:
        if file in npz_files_removed:
            continue

        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        shutil.copy2(src_path, dst_path)

    return f"Copied .npz files from {src} to {dst} with {removal_percentage}% removed."


subset_percentage = 10
src = "/home/george-vengrovski/Documents/data/llb3_data_matrices"
dst = f"/home/george-vengrovski/Documents/data/llb3_{str(subset_percentage)}%_subset"

copy_folder_with_removal(src, dst, subset_percentage)
