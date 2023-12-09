import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("src")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_cluster/project')

# For removing mat files 
folders = ['/home/george-vengrovski/Documents/data/llb3_data_matrices', '/home/george-vengrovski/Documents/data/llb11_data_matrices', '/home/george-vengrovski/Documents/data/llb16_data_matrices']

for folder in folders:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith('.mat'):
                os.remove(os.path.join(folder, filename))
                print(f'Removed {filename} from {folder}')
    else:
        print(f'Folder does not exist: {folder}')

