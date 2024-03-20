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
import itertools
import csv
import sys
import multiprocessing
import soundfile as sf

class WavtoSpec:
    def __init__(self, src_dir, dst_dir, csv_file_dir=None):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.csv_file_dir = csv_file_dir
        self.use_csv = csv_file_dir is not None

    def process_file(self, file_path, song_info):
        self.convert_to_spectrogram(file_path, song_info)

    def process_directory(self):
        if self.use_csv:
            song_info = self.read_csv_file(self.csv_file_dir)  # Read the CSV file and store song information
        else:
            song_info = {}  # Empty dictionary when not using CSV file

        # Get the list of audio files
        audio_files = []
        for root, dirs, files in os.walk(self.src_dir):
            dirs[:] = [d for d in dirs if d not in ['.DS_Store']]  # Ignore irrelevant directories
            audio_files.extend([os.path.join(root, file) for file in files if file.lower().endswith('.wav')])

        # Create a pool of worker processes
        num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
        pool = multiprocessing.Pool(processes=num_processes)

        # Create a progress bar
        progress_bar = tqdm(total=len(audio_files), unit='file', desc='Processing files')

        # Distribute the work among the processes
        results = []
        for file_path in audio_files:
            result = pool.apply_async(self.process_file, args=(file_path, song_info))
            results.append(result)

        # Update the progress bar as files are processed
        for result in results:
            result.get()
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Close the pool
        pool.close()
        pool.join()

    def read_csv_file(self, csv_file):
        song_info = {}
        csv.field_size_limit(sys.maxsize)  # Increase the field size limit
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                song_name = row['song_name']
                song_ms = eval(row['song_ms'])  # Evaluate the string to get the list of tuples
                song_info[song_name] = song_ms
        return song_info

    def convert_to_spectrogram(self, file_path, song_info, min_length_ms=1000, default_sample_rate=44100, max_timebins=5000):
        offset_constant = .001
        try:
            # Open the audio file using soundfile
            with sf.SoundFile(file_path, 'r') as audio_file:
                samplerate = audio_file.samplerate
                chunk_size = 1000000  # Adjust the chunk size as needed

                # Process the audio file in chunks
                data = []
                while True:
                    chunk = audio_file.read(chunk_size)
                    if len(chunk) == 0:
                        break
                    data.append(chunk)

                data = np.concatenate(data)

            # Check if the data is multichannel (e.g., stereo)
            if len(data.shape) > 1:
                data = data[:, 1]  # Select the right channel, channel 2

            # Resample the data if necessary
            if samplerate != default_sample_rate:
                num_samples = int(len(data) * default_sample_rate / samplerate)
                data = resample(data, num_samples)
                samplerate = default_sample_rate

            NFFT = 1024  
            step_size = 512 

            overlap_samples = NFFT - step_size

            song_name = os.path.splitext(os.path.basename(file_path))[0]
            print(song_name)

            if self.use_csv and song_name in song_info:
                print("found_song")
                song_segments = song_info[song_name]
                for segment in song_segments:
                    start_ms, end_ms = segment
                    start_sample = int(start_ms * samplerate / 1000)
                    end_sample = int(end_ms * samplerate / 1000)
                    segment_data = data[start_sample:end_sample]

                    # Generate spectrogram for the segment
                    f, t, Sxx = spectrogram(segment_data, fs=samplerate, nperseg=NFFT, noverlap=overlap_samples)

                    # Convert to dB and apply z-scoring
                    Sxx_log = 10 * np.log10(Sxx + offset_constant)

                    mean = Sxx_log.mean()
                    std = Sxx_log.std()
                    Sxx_z_scored = (Sxx_log - mean) / std

                    # Check if the spectrogram exceeds the max_timebins limit
                    if Sxx_z_scored.shape[1] > max_timebins:
                        # Calculate the number of parts needed
                        num_parts = int(np.ceil(Sxx_z_scored.shape[1] / max_timebins))
                        for part in range(num_parts):
                            start_bin = part * max_timebins
                            end_bin = min((part + 1) * max_timebins, Sxx_z_scored.shape[1])
                            spec_part = Sxx_z_scored[:, start_bin:end_bin]

                            # Define modified title for each part
                            segment_filename = f"{song_name}_{int(start_ms)}_{int(end_ms)}_part{part+1}"
                            segment_file_path = os.path.join(self.dst_dir, segment_filename + '.npz')

                            # Save each part
                            np.savez_compressed(segment_file_path, s=spec_part)
                    else:
                        # If the spectrogram does not exceed the limit, save it normally
                        segment_filename = f"{song_name}_{int(start_ms)}_{int(end_ms)}"
                        segment_file_path = os.path.join(self.dst_dir, segment_filename + '.npz')
                        np.savez_compressed(segment_file_path, s=Sxx_z_scored)
            elif not self.use_csv:
                # If use_csv is False or the song is not found in the CSV file, treat the entire WAV file as a song segment
                print(f"Treating the entire WAV file as a song segment: {song_name}")

                # Generate spectrogram for the entire WAV file
                f, t, Sxx = spectrogram(data, fs=samplerate, nperseg=NFFT, noverlap=overlap_samples)

                # Convert to dB and apply z-scoring
                Sxx_log = 10 * np.log10(Sxx + offset_constant)

                mean = Sxx_log.mean()
                std = Sxx_log.std()
                Sxx_z_scored = (Sxx_log - mean) / std

                # Check if the spectrogram exceeds the max_timebins limit
                if Sxx_z_scored.shape[1] > max_timebins:
                    # Calculate the number of parts needed
                    num_parts = int(np.ceil(Sxx_z_scored.shape[1] / max_timebins))
                    for part in range(num_parts):
                        start_bin = part * max_timebins
                        end_bin = min((part + 1) * max_timebins, Sxx_z_scored.shape[1])
                        spec_part = Sxx_z_scored[:, start_bin:end_bin]

                        # Define modified title for each part
                        segment_filename = f"{song_name}_part{part+1}"
                        segment_file_path = os.path.join(self.dst_dir, segment_filename + '.npz')

                        # Save each part
                        np.savez_compressed(segment_file_path, s=spec_part)
                else:
                    # If the spectrogram does not exceed the limit, save it normally
                    segment_filename = f"{song_name}"
                    segment_file_path = os.path.join(self.dst_dir, segment_filename + '.npz')
                    np.savez_compressed(segment_file_path, s=Sxx_z_scored)

            else:
                print("no song found, skipping file")

            plt.close()

        except ValueError as e:
            print(f"Error reading {file_path}: {e}")
        except MemoryError as e:
            print(f"Memory error occurred while processing {file_path}: {e}")
            print("Skipping this file and moving on to the next.")

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

    def plot_grid_of_spectrograms(self, min_length=500):
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
                if spectrogram_data.shape[1] > min_length:  # Check if it has more than 2000 time bins
                    selected_spec_paths.append(spec_path)  # Add to the list if it meets the criteria

        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
        fig.suptitle('5 x 5 Grid of Random Spectrograms', fontsize=16)

        for ax, spec_path in zip(axes.flatten(), selected_spec_paths):
            with np.load(spec_path) as data:
                spectrogram_data = data['s']
                # Take the second set of 1000 bins
                spectrogram_data = spectrogram_data[:, :min_length]

                # Plot the spectrogram on its respective subplot
                ax.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(spec_path.stem, fontsize=8)
                ax.axis('off')  # Hide axes for better visualization

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def compare_spectrogram_permutations(self, param_dict):
        """
        Generates and plots spectrograms for a single random .wav file using permutations of NFFT and step_size parameters.
        Only the first 1000 time bins of each spectrogram are plotted, with the row parameters (NFFT and step size)
        displayed on the left-hand side of each row.

        Parameters:
        param_dict (dict): A dictionary with 'NFFT' and 'step_size' as keys and lists of their possible values.
        """
        # Randomly select 1 .wav file
        wav_files = [f for f in Path(self.src_dir).glob('**/*.wav')]
        selected_file = random.choice(wav_files)

        # Generate all permutations of NFFT and step_size
        permutations = list(itertools.product(param_dict['NFFT'], param_dict['step_size']))

        # Set up the plot
        fig, axs = plt.subplots(len(permutations), 1, figsize=(10, 5 * len(permutations)), squeeze=False)
        
        for i, (nfft, step_size) in enumerate(permutations):
            # Generate the spectrogram for the current permutation and file
            spectrogram_data = self.generate_spectrogram_data(str(selected_file), nfft, step_size)
            spectrogram_data = spectrogram_data[:, :1000]  # Select the first 1000 time bins

            # Plot the spectrogram
            ax = axs[i, 0]
            img = ax.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='viridis')
            ax.set_ylabel(f'NFFT: {nfft}\nStep Size: {step_size}')
            ax.set_xlabel('Time Bins')
            ax.axis('on')

            # Adding a colorbar to each subplot
            fig.colorbar(img, ax=ax)

        plt.tight_layout()
        plt.show()

    def generate_spectrogram_data(self, file_path, nfft, step_size):
        """
        Generates spectrogram data for a given .wav file, NFFT, and step size.

        Parameters:
        file_path (str): Path to the .wav file.
        nfft (int): Number of FFT points.
        step_size (int): Step size for the FFT overlap.

        Returns:
        ndarray: The spectrogram data.
        """
        # This is a simplified version of convert_to_spectrogram method
        samplerate, data = wavfile.read(file_path)
        if len(data.shape) > 1:
            data = data[:, 1]  # Select the right channel if stereo

        # # Use a Gaussian window
        # window = windows.gaussian(nfft, std=nfft/8)

        # Calculate the overlap in samples
        overlap_samples = nfft - step_size

        # Compute the spectrogram with the Gaussian window
        f, t, Sxx = spectrogram(data, fs=samplerate, nperseg=nfft, noverlap=overlap_samples)

        # Convert to dB
        Sxx_log = 10 * np.log10(Sxx + .001)

        mean = Sxx_log.mean()
        std = Sxx_log.std()
        Sxx_z_scored = (Sxx_log - mean) / std

        return Sxx_z_scored

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
# csv_dir only populated if u want to use it 
wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/zebra_finch/combined_wav', '/media/george-vengrovski/disk2/zebra_finch/combined_spec')
wav_to_spec.process_directory()
# # # wav_to_spec.analyze_dataset()
# wav_to_spec.plot_grid_of_spectrograms()



# param_dict = {
#     'NFFT': [512, 1024], 
#     'step_size': [256, 512] 
# }
# wav_to_spec.compare_spectrogram_permutations(param_dict)