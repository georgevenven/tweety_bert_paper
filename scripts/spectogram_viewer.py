
import numpy as np
import matplotlib.pyplot as plt

file = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/combined_spectrogram.npz"


def plot_spectrogram(file_path):
    # Load the .npz file
    data = np.load(file_path)
    spectrogram = data['s']

    # Plot the spectrogram
    plt.figure(figsize=(12, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Intensity')
    plt.show()


plot_spectrogram(file)
