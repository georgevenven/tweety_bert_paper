import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
from tqdm import tqdm

def copy_random_files(src, dst, num_files=1000):
    """
    Randomly copies a specified number of .npz files from `src` to `dst`.

    :param src: Source directory path.
    :param dst: Destination directory path.
    :param num_files: Number of .npz files to be copied. Default is 1000.
    """

    if not os.path.exists(src):
        raise ValueError(f"Source directory {src} does not exist.")

    if os.path.exists(dst):
        raise ValueError(f"Destination directory {dst} already exists.")

    # Create destination directory
    os.makedirs(dst)

    # List all npz files in the source directory
    npz_files = [item for item in os.listdir(src) if item.endswith(".npz")]

    # If there are fewer npz files than requested, copy all of them
    if len(npz_files) < num_files:
        print(f"Only {len(npz_files)} files found. Copying all of them.")
        num_files = len(npz_files)

    # Randomly select npz files to copy
    npz_files_to_copy = set(random.sample(npz_files, num_files))

    # Copy selected npz files and display progress
    for file in tqdm(npz_files_to_copy):
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        shutil.copy2(src_path, dst_path)

    return f"Copied {num_files} .npz files from {src} to {dst}."

# Usage
subset_number = 10000
src = "/home/george-vengrovski/Documents/data/pretrain_dataset"
dst = "/home/george-vengrovski/Documents/data/pretrain_songdetector"

copy_random_files(src, dst, subset_number)
