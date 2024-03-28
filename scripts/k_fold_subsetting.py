import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import KFold

def k_fold_validation(src, params):
    """
    Generates folders for k-fold validation on the specified dataset.
    :param src: Source directory path.
    :param params: Dictionary containing the following keys:
                   - 'k': Number of folds (default: 5).
                   - 'subset_percentages': List of subset percentages to evaluate (default: [0.1]).
    """
    if not os.path.exists(src):
        raise ValueError(f"Source directory {src} does not exist.")

    # Set default values if not provided in params
    k = params.get('k', 5)
    subset_percentages = params.get('subset_percentages', [0.1])

    # List all npz files in the source directory
    npz_files = [item for item in os.listdir(src) if item.endswith(".npz")]

    # Create k-fold split
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Generate folders for each subset percentage
    for subset_percentage in subset_percentages:
        print(f"Subset Percentage: {subset_percentage}")

        # Calculate the number of files to use based on the given percentage
        num_files = int(len(npz_files) * subset_percentage)

        # If there are no files to use, skip this subset percentage
        if num_files == 0:
            print("The subset percentage results in zero files to be used. Skipping.")
            continue

        for fold, (train_indices, val_indices) in enumerate(kf.split(npz_files), 1):
            print(f"Fold {fold}")

            # Create train and validation sets
            train_files = [npz_files[i] for i in train_indices]
            val_files = [npz_files[i] for i in val_indices]

            # Create train and validation directories
            train_dir = f"files/llb3_train_{subset_percentage}_fold{fold}_train"
            val_dir = f"files/llb3_train_{subset_percentage}_fold{fold}_val"

            # Copy train files to train directory
            os.makedirs(train_dir, exist_ok=True)
            for file in tqdm(train_files, desc="Copying train files"):
                src_path = os.path.join(src, file)
                dst_path = os.path.join(train_dir, file)
                shutil.copy2(src_path, dst_path)

            # Copy validation files to validation directory
            os.makedirs(val_dir, exist_ok=True)
            for file in tqdm(val_files, desc="Copying validation files"):
                src_path = os.path.join(src, file)
                dst_path = os.path.join(val_dir, file)
                shutil.copy2(src_path, dst_path)

# Usage
src = "files/llb3_train"
params = {
    'k': 5,
    'subset_percentages': [0.01, 0.1]
}
k_fold_validation(src, params)