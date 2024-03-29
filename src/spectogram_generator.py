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
import wave
import gc
from memory_profiler import profile
import psutil

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Memory usage in MB

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

    #@profile
    def convert_to_spectrogram(self, file_path, song_info, max_timebins=5000, chunk_size=52428800):
        offset_constant = .001
        # print(f"Memory usage before processing: {get_memory_usage()} MB")

        try:
            with sf.SoundFile(file_path, 'r') as wav_file:
                samplerate = wav_file.samplerate
                num_channels = wav_file.channels

                # Process the audio data in chunks to reduce memory usage
                data = []
                while True:
                    chunk = wav_file.read(chunk_size)
                    if len(chunk) == 0:
                        break
                    # Normalize and append chunk data
                    chunk = (chunk * 32767).astype(np.int16)
                    if num_channels > 1:
                        chunk = chunk[:, 0]  # Use the first channel if multi-channel audio
                    data.append(chunk)
                data = np.concatenate(data)  # Combine chunks into a single array

                ## This causes memory leaks!! uncomment only if problem is understood 
                # # Resample if necessary
                # if samplerate != default_sample_rate:
                #     num_samples = int(len(data) * default_sample_rate / samplerate)
                #     data = resample(data, num_samples)
                #     samplerate = default_sample_rate

                song_name = os.path.splitext(os.path.basename(file_path))[0]

                # Define spectrogram parameters
                NFFT = 1024  
                step_size = 512
                overlap_samples = NFFT - step_size

                segments_to_process = self.get_segments_to_process(song_name, song_info, data, samplerate)

                for start_sample, end_sample in segments_to_process:
                    segment_data = data[start_sample:end_sample]
                    # Generate and process spectrogram for the segment
                    f, t, Sxx = spectrogram(segment_data, fs=samplerate, nperseg=NFFT, noverlap=overlap_samples)
                    Sxx_log = 10 * np.log10(Sxx + offset_constant)
                    Sxx_z_scored = self.z_score_spectrogram(Sxx_log)

                    self.save_spectrogram_parts(song_name, Sxx_z_scored, start_sample, samplerate, max_timebins=max_timebins)

                    # Delete intermediate arrays
                    del segment_data, Sxx, Sxx_log, Sxx_z_scored

                # Delete data array
                del data

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        finally:
            plt.close('all')  # Ensure all plots are closed
            gc.collect()  # Force garbage collection
            # print(f"Memory usage after processing: {get_memory_usage()} MB")

    #@profile
    def get_segments_to_process(self, song_name, song_info, data, samplerate):
        segments_to_process = []
        if self.use_csv and song_name in song_info:
            song_segments = song_info[song_name]
            for start_ms, end_ms in song_segments:
                start_sample = int(start_ms * samplerate / 1000)
                end_sample = int(end_ms * samplerate / 1000)
                segments_to_process.append((start_sample, end_sample))
        else:
            # If not using CSV, or song not found, process the entire file as one segment
            segments_to_process.append((0, len(data)))
        return segments_to_process

    def z_score_spectrogram(self, Sxx_log):
        mean = Sxx_log.mean()
        std = Sxx_log.std()
        Sxx_z_scored = (Sxx_log - mean) / std
        return Sxx_z_scored

    #@profile
    def save_spectrogram_parts(self, song_name, Sxx_z_scored, start_sample, samplerate, max_timebins=5000):
        if Sxx_z_scored.shape[1] > max_timebins:
            num_parts = int(np.ceil(Sxx_z_scored.shape[1] / max_timebins))
            for part in range(num_parts):
                start_bin = part * max_timebins
                end_bin = min((part + 1) * max_timebins, Sxx_z_scored.shape[1])
                spec_part = Sxx_z_scored[:, start_bin:end_bin]
                segment_filename = f"{song_name}_{int(start_sample / samplerate * 1000)}_part{part+1}.npz"
                segment_file_path = os.path.join(self.dst_dir, segment_filename)
                np.savez_compressed(segment_file_path, s=spec_part)
        else:
            segment_filename = f"{song_name}_{int(start_sample / samplerate * 1000)}.npz"
            segment_file_path = os.path.join(self.dst_dir, segment_filename)
            np.savez_compressed(segment_file_path, s=Sxx_z_scored)

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

    def plot_grid_of_spectrograms(self, min_length=100):
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
                if spectrogram_data.shape[1] > min_length:  # Check if it has more than min_length time bins
                    selected_spec_paths.append(spec_path)  # Add to the list if it meets the criteria

        # Set up the subplot grid
        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
        fig.suptitle('5 x 5 Grid of Random Spectrograms', fontsize=16)

        for ax, spec_path in zip(axes.flatten(), selected_spec_paths):
            with np.load(spec_path) as data:
                spectrogram_data = data['s']
                # Take the first set of min_length bins
                spectrogram_data = spectrogram_data[:, :min_length]

                # Plot the spectrogram on its respective subplot
                img = ax.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(spec_path.stem, fontsize=8)
                ax.axis('off')  # Hide axes for better visualization

                # Create a color bar for the current subplot
                fig.colorbar(img, ax=ax, format='%+2.0f dB')

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

if __name__ == '__main__':
    # wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/brown_thrasher/brown_thrasher_wav', '/media/george-vengrovski/disk2/brown_thrasher/brown_thrasher_specs')
    # wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/zebra_finch/combined_wav', '/media/george-vengrovski/disk2/zebra_finch/combined_specs')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/canary_yarden/combined_wav', '/media/george-vengrovski/disk2/canary_yarden/combined_no_clip_specs')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/canary/sorted_2/combined_wav', '/media/george-vengrovski/disk2/canary/sorted_2/combined_specs')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/canary/sorted_1/combined_wav', '/media/george-vengrovski/disk2/canary/sorted_1/combined_spec')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/budgie/warble_wav', '/media/george-vengrovski/disk2/budgie/warble_spec')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/budgie/T5_ssd_combined', '/media/george-vengrovski/disk2/budgie/T5_ssd_combined_specs')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/budgie/pair_wav', '/media/george-vengrovski/disk2/budgie/pair_spec')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/brown_thrasher/brown_thrasher_wav', '/media/george-vengrovski/disk2/brown_thrasher/brown_thrasher_specs')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/bengalese-finch/bengalese-finch_nickle_dave/combined_wav', '/media/george-vengrovski/disk2/bengalese-finch/bengalese-finch_nickle_dave/combined_specs')
    wav_to_spec.process_directory()
    wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/bengalese-finch/3470165/combined_wav', '/media/george-vengrovski/disk2/bengalese-finch/3470165/combined_specs')
    wav_to_spec.process_directory()

    # # Usage:
    # csv_dir only populated if u want to use it 

    # # # wav_to_spec.analyze_dataset()
    # wav_to_spec.plot_grid_of_spectrograms()



    # param_dict = {
    #     'NFFT': [512, 1024], 
    #     'step_size': [256, 512] 
    # }
    # wav_to_spec.compare_spectrogram_permutations(param_dict)