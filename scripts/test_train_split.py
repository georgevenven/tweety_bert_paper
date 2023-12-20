import os
import shutil
import random

def split_dataset(folder_path, train_ratio, train_folder_dest, test_folder_dest):
    """
    Splits the npz files in the given folder into train and test sets based on the specified ratio
    and copies them to specified train and test destination folders.

    Parameters:
    folder_path (str): The path to the folder containing the dataset.
    train_ratio (float): The ratio of npz files to be included in the train set.
    train_folder_dest (str): The path to the destination train folder.
    test_folder_dest (str): The path to the destination test folder.
    """
    # Create train and test directories in the specified destination folders
    os.makedirs(train_folder_dest, exist_ok=True)
    os.makedirs(test_folder_dest, exist_ok=True)

    # List all npz files in the source folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.npz')]

    # Shuffle the files
    random.shuffle(all_files)

    # Calculate number of files for the train set
    train_size = int(len(all_files) * train_ratio)

    # Split files
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    # Copy files to respective destination directories
    for file in train_files:
        shutil.copy2(os.path.join(folder_path, file), os.path.join(train_folder_dest, file))

    for file in test_files:
        shutil.copy2(os.path.join(folder_path, file), os.path.join(test_folder_dest, file))

# Example usage
split_dataset('/home/george-vengrovski/Documents/data/llb3_data_matrices', 0.8, '/home/george-vengrovski/Documents/projects/tweety_bert_paper/llb3_train', '/home/george-vengrovski/Documents/projects/tweety_bert_paper/llb3_test')
split_dataset('/home/george-vengrovski/Documents/data/llb16_data_matrices', 0.8, '/home/george-vengrovski/Documents/projects/tweety_bert_paper/llb16_train', '/home/george-vengrovski/Documents/projects/tweety_bert_paper/llb16_test')
split_dataset('/home/george-vengrovski/Documents/data/llb11_data_matrices', 0.8, '/home/george-vengrovski/Documents/projects/tweety_bert_paper/llb11_train', '/home/george-vengrovski/Documents/projects/tweety_bert_paper/llb11_test')
