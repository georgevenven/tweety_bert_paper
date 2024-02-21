import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows, spectrogram, ellip, filtfilt
import shutil
from pathlib import Path


class WavtoSpec:
    def __init__(self, src_dir, dst_dir):
        self.src_dir = src_dir
        self.dst_dir = dst_dir

    def process_directory(self):
        # First walk to count all the .wav files
        total_files = sum(
            len([f for f in files if f.lower().endswith('.wav')])
            for r, d, files in os.walk(self.src_dir)
        )
        
        # Now process each file with a single tqdm bar
        with tqdm(total=total_files, desc="Overall progress") as pbar:
            for root, dirs, files in os.walk(self.src_dir):
                dirs[:] = [d for d in dirs if d not in ['.DS_Store']]  # Ignore irrelevant directories
                files = [f for f in files if f.lower().endswith('.wav')]
                for file in files:
                    full_path = os.path.join(root, file)
                    self.convert_to_spectrogram(full_path)
                    pbar.update(1)  # Update the progress bar for each file
    
    def convert_to_spectrogram(self, file_path, min_length_ms=1000):
        try:
            samplerate, data = wavfile.read(file_path)
            FS = samplerate # input
            NFFT = 512
            noverlap = 450  # noverlap > NFFT/2
            # Create Spectrogram
            spectrum, freqs, t, im = plt.specgram(data, NFFT=NFFT, Fs=FS, noverlap=noverlap,cmap='jet')
            # Manual Params (can be changed)
            logThresh = 0
            afterThresh = 0
            # Take log then delete elements below another thresh after log
            #filterSpec = spectrum
            filterSpec = np.log(spectrum + logThresh)
            filterSpec[np.where(filterSpec < afterThresh)] = 0
            colormap = 'jet'
            # Normalize the numeric array to the [0, 1] range
            normalized_array = (filterSpec - np.min(filterSpec)) / (np.max(filterSpec) - np.min(filterSpec))
            
            # Assuming label is an integer or float
            labels = np.full((normalized_array.shape[1],), 0)  # Adjust the label array as needed

            # Define the path where the spectrogram will be saved
            spec_filename = os.path.splitext(os.path.basename(file_path))[0]
            spec_file_path = os.path.join(self.dst_dir, spec_filename + '.npz')

            # Saving the spectrogram and the labels
            np.savez_compressed(spec_file_path, s=normalized_array, labels=labels)

            # Print out the path to the saved file
            print(f"Spectrogram saved to {spec_file_path}")
            plt.close()

        except ValueError as e:
            print(f"Error reading {file_path}: {e}")


    def visualize_random_spectrogram(self):
        # Get a list of all '.npz' files in the destination directory
        npz_files = list(Path(self.dst_dir).glob('*.npz'))
        if not npz_files:
            print("No spectrograms available to visualize.")
            return
        
        # Choose a random spectrogram file
        random_spec_path = random.choice(npz_files)
        
        # Load the spectrogram data from the randomly chosen file
        with np.load(random_spec_path) as data:
            spectrogram_data = data['s']
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_data, aspect='auto', origin='lower')
        plt.title(f"Random Spectrogram: {random_spec_path.stem}")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

# Helper function to find the next power of two
def nextpow2(x):
    return np.ceil(np.log2(np.abs(x))).astype('int')

def copy_yarden_data(src_dirs, dst_dir):
    """
    Copies all .npz files from a list of source directories to a destination directory.

    Parameters:
    src_dirs (list): A list of source directories to search for .npz files.
    dst_dir (str): The destination directory where .npz files will be copied.
    """
    # Ensure the destination directory exists
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    # Create a list to store all the .npz files found
    npz_files = []

    # Find all .npz files in source directories
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append((os.path.join(root, file), file))

    # Copy the .npz files to the destination directory with progress bar
    for src_file_path, file in tqdm(npz_files, desc='Copying files'):
        dst_file_path = os.path.join(dst_dir, file)
        
        # Ensure we don't overwrite files in the destination
        if os.path.exists(dst_file_path):
            print(f"File {file} already exists in destination. Skipping copy.")
            continue

        # Copy the .npz file to the destination directory
        shutil.copy2(src_file_path, dst_file_path)
        print(f"Copied {file} to {dst_dir}")


# Usage:
wav_to_spec = WavtoSpec('/home/george-vengrovski/Documents/data/song_detector_test', '/home/george-vengrovski/Documents/data/temp')
wav_to_spec.process_directory()
# wav_to_spec.save_npz()
# wav_to_spec.visualize_random_spectrogram()
