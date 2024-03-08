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
from scipy.signal import resample

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
    
    def convert_to_spectrogram(self, file_path, min_length_ms=1000, default_sample_rate=44100):
        try:
            samplerate, data = wavfile.read(file_path)

            # Check if the data is multichannel (e.g., stereo)
            if len(data.shape) > 1:
                # Select one channel (e.g., the first channel)
                data = data[:, 0]

            # Resample the data if the samplerate does not match the default_sample_rate
            if samplerate != default_sample_rate:
                # print(f"Resampling from {samplerate}Hz to {default_sample_rate}Hz.")
                # Calculate the number of samples after resampling
                num_samples = int(len(data) * default_sample_rate / samplerate)
                # Resample the data
                data = resample(data, num_samples)
                # Update the samplerate to the default_sample_rate
                samplerate = default_sample_rate

            # Calculate the length of the audio file in milliseconds
            length_in_ms = (data.shape[0] / samplerate) * 1000

            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return  # Skip processing this file

            # High-pass filter (adjust the filtering frequency as necessary)
            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)

            # Canary song analysis parameters
            NFFT = 1024  # Number of points in FFT
            step_size = 119  # Step size for overlap

            # Calculate the overlap in samples
            overlap_samples = NFFT - step_size

            # Use a Gaussian window
            window = windows.gaussian(NFFT, std=NFFT/8)

            # Compute the spectrogram with the Gaussian window
            f, t, Sxx = spectrogram(data, fs=samplerate, window=window, nperseg=NFFT, noverlap=overlap_samples)

            # Convert to dB
            Sxx_log = 10 * np.log10(Sxx)

            # # Post-processing: Clipping and Normalization
            # clipping_level = -2  # dB
            # clipped_spec = np.clip(Sxx_log, a_min=clipping_level, a_max=None)
           
            mean = Sxx_log.mean()
            std = Sxx_log.std()

            z_scored_spec = (Sxx_log - mean) / std
            
            # Define the path where the spectrogram will be saved
            spec_filename = os.path.splitext(os.path.basename(file_path))[0]
            spec_file_path = os.path.join(self.dst_dir, spec_filename + '.npz')

            # Saving the spectrogram and the labels
            np.savez_compressed(spec_file_path, s=z_scored_spec)
            plt.close()

        except ValueError as e:
            print(f"Error reading {file_path}: {e}")

    # def convert_to_spectrogram(self, file_path, min_length_ms=1000):
    #     try:
    #         samplerate, data = wavfile.read(file_path)
    #         FS = samplerate # input
    #         NFFT = 512
    #         noverlap = 450  # noverlap > NFFT/2
    #         # Create Spectrogram
    #         spectrum, freqs, t, im = plt.specgram(data, NFFT=NFFT, Fs=FS, noverlap=noverlap,cmap='jet')
    #         # Manual Params (can be changed)
    #         logThresh = 0
    #         afterThresh = 0
    #         # Take log then delete elements below another thresh after log
    #         #filterSpec = spectrum
    #         filterSpec = np.log(spectrum + logThresh)
    #         filterSpec[np.where(filterSpec < afterThresh)] = 0
    #         # Normalize the numeric array to the [0, 1] range
    #         normalized_array = (filterSpec - np.min(filterSpec)) / (np.max(filterSpec) - np.min(filterSpec))
    #         print(normalized_array.shape)
    #         # Apply the colormap to the normalized array
    #         # rgb_array = plt.cm.get_cmap(colormap)(normalized_array)
    #         # # Read the WAV file
    #         # samplerate, data = wavfile.read(file_path)
    #         # # Calculate the length of the audio file in milliseconds
    #         # length_in_ms = (data.shape[0] / samplerate) * 1
    #         # Assuming label is an integer or float
    #         labels = np.full((normalized_array.shape[1],), 0)  # Adjust the label array as needed
    #         # Define the path where the spectrogram will be saved
    #         spec_filename = os.path.splitext(os.path.basename(file_path))[0]
    #         spec_file_path = os.path.join(self.dst_dir, spec_filename + '.npz')
    #         # Saving the spectrogram and the labels
    #         np.savez_compressed(spec_file_path, s=normalized_array, labels=labels)
    #         # Print out the path to the saved file
    #         print(f"Spectrogram saved to {spec_file_path}")
    #     except ValueError as e:
    #         print(f"Error reading {file_path}: {e}")   

    #     except ValueError as e:
    #         print(f"Error reading {file_path}: {e}")


    def analyze_dataset(self, min_length_ms=1000, default_sample_rate=2e4):
        raw_means, raw_stds = [], []
        spec_means, spec_stds = [], []

        total_files = sum(1 for _, _, files in os.walk(self.src_dir) for file in files if file.lower().endswith('.wav'))

        with tqdm(total=total_files, desc="Analyzing WAV files") as pbar:
            for root, dirs, files in os.walk(self.src_dir):
                for file in files:
                    if file.lower().endswith('.wav'):
                        file_path = os.path.join(root, file)
                        try:
                            samplerate, data = wavfile.read(file_path)
                            if len(data.shape) > 1:
                                data = data[:, 0]
                            if samplerate != default_sample_rate:
                                num_samples = int(len(data) * default_sample_rate / samplerate)
                                data = resample(data, num_samples)
                                samplerate = default_sample_rate

                            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
                            filtered_data = filtfilt(b, a, data)
                            window = windows.gaussian(1024, std=1024/8)
                            f, t, Sxx = spectrogram(filtered_data, fs=samplerate, window=window, nperseg=1024, noverlap=1024 - 119)
                            Sxx_log = 10 * np.log10(Sxx)
                            mean, std = Sxx_log.mean(), Sxx_log.std()

                            # Store means and stds for each file
                            raw_means.append(np.mean(data))
                            raw_stds.append(np.std(data))
                            spec_means.append(mean)
                            spec_stds.append(std)

                        except ValueError as e:
                            print(f"Error reading {file_path}: {e}")
                        finally:
                            pbar.update(1)

        # Plotting for means
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Raw values mean plot
        axs[0, 0].hist(raw_means, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axs[0, 0].set_title('Distribution of Raw Audio Means')
        axs[0, 0].set_xlabel('Mean Value')
        axs[0, 0].set_ylabel('Frequency')

        # Spectrogram values mean plot
        axs[1, 0].hist(spec_means, bins=30, color='green', alpha=0.7, edgecolor='black')
        axs[1, 0].set_title('Distribution of Spectrogram Means')
        axs[1, 0].set_xlabel('Mean Value')
        axs[1, 0].set_ylabel('Frequency')

        # Plotting for standard deviations
        # Raw values std plot
        axs[0, 1].hist(raw_stds, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axs[0, 1].set_title('Distribution of Raw Audio Standard Deviations')
        axs[0, 1].set_xlabel('Standard Deviation')
        axs[0, 1].set_ylabel('Frequency')

        # Spectrogram values std plot
        axs[1, 1].hist(spec_stds, bins=30, color='green', alpha=0.7, edgecolor='black')
        axs[1, 1].set_title('Distribution of Spectrogram Standard Deviations')
        axs[1, 1].set_xlabel('Standard Deviation')
        axs[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

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
        plt.colorbar(format='%+2.0f Z Scores')
        plt.show()

    def plot_grid_of_spectrograms(self):
        # Get a list of all '.npz' files in the destination directory
        npz_files = list(Path(self.dst_dir).glob('*.npz'))
        if len(npz_files) < 25:
            print("Not enough spectrograms available to create a 5x5 grid. Please ensure there are at least 25 spectrograms in the destination directory.")
            return

        selected_spec_paths = []
        attempted_paths = set()

        # Keep selecting random spectrograms until we have 25 with enough time bins
        while len(selected_spec_paths) < 25:
            if len(attempted_paths) >= len(npz_files):
                print("Tried all available spectrograms, but could not find enough with sufficient time bins.")
                return
            spec_path = random.choice(npz_files)
            if spec_path in attempted_paths:
                continue  # Skip if this spectrogram has already been attempted
            attempted_paths.add(spec_path)

            with np.load(spec_path) as data:
                spectrogram_data = data['s']
                if spectrogram_data.shape[1] > 2000:  # Check if it has more than 2000 time bins
                    selected_spec_paths.append(spec_path)  # Add to the list if it meets the criteria

        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
        fig.suptitle('5 x 5 Grid of Random Spectrograms', fontsize=16)

        for ax, spec_path in zip(axes.flatten(), selected_spec_paths):
            with np.load(spec_path) as data:
                spectrogram_data = data['s']
                # Take the second set of 1000 bins
                spectrogram_data = spectrogram_data[:, 1000:2000]

                # Plot the spectrogram on its respective subplot
                ax.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(spec_path.stem, fontsize=8)
                ax.axis('off')  # Hide axes for better visualization

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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


# # Usage:
wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/budgie/dev_wav', '/media/george-vengrovski/disk2/budgie/dev_npz_carrot_method')
wav_to_spec.process_directory()
# wav_to_spec.analyze_dataset()
wav_to_spec.plot_grid_of_spectrograms()
# # wav_to_spec.save_npz()
# # wav_to_spec.visualize_random_spectrogram()
