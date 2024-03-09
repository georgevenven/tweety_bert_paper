import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
from tqdm import tqdm

def copy_random_files(src, dst, subset_percentage=0.1):
    """
    Randomly copies a specified percentage of .npz files from `src` to `dst`.

    :param src: Source directory path.
    :param dst: Destination directory path.
    :param subset_percentage: Percentage of .npz files to be copied, e.g., 0.1 for 10%. Default is 0.1.
    """

    if not os.path.exists(src):
        raise ValueError(f"Source directory {src} does not exist.")

    if os.path.exists(dst):
        raise ValueError(f"Destination directory {dst} already exists.")

    # Create destination directory
    os.makedirs(dst)

    # List all npz files in the source directory
    npz_files = [item for item in os.listdir(src) if item.endswith(".npz")]

    # Calculate the number of files to copy based on the given percentage
    num_files = int(len(npz_files) * subset_percentage)

    # If there are no files to copy, exit the function
    if num_files == 0:
        return "The subset percentage results in zero files to be copied. Please increase the subset percentage."

    # Randomly select npz files to copy
    npz_files_to_copy = set(random.sample(npz_files, num_files))

    # Copy selected npz files and display progress
    for file in tqdm(npz_files_to_copy):
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        shutil.copy2(src_path, dst_path)

    return f"Copied {num_files} (.npz) files from {src} to {dst}."

# Usage
subset_percentage = 0.01  # 10% of the total files
src = "files/llb3_train"
dst = "files/llb3_train_1_precent"
copy_random_files(src, dst, subset_percentage)
