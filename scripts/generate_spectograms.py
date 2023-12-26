import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("src")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

from spectogram_generator import WavtoSpec, copy_yarden_data

# # # # For generating specs from Rose's dataset
# wav_to_spec = WavtoSpec('/home/george-vengrovski/Documents/data/raw_canary_recordings', '/home/george-vengrovski/Documents/data/pretrain_dataset')
# wav_to_spec.process_directory()


# For copying Yardens data into one master location
copy_yarden_data(['/home/george-vengrovski/Documents/data/llb3_data_matrices', '/home/george-vengrovski/Documents/data/llb11_data_matrices', '/home/george-vengrovski/Documents/data/llb16_data_matrices'], '/home/george-vengrovski/Documents/data/pretrain_dataset')
