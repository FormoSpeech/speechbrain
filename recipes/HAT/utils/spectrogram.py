import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Define the file names
wav_files = [
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_android.wav',
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_iOS.wav',
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_PCmic.wav',
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_webcam.wav',
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_condenser.wav',
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_lavalier.wav',
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_H8x.wav',
    '/mnt/user_forbes/datasets/sixian_reading/train/XF0050001A2033_1_channel_H8y.wav'
]

# Create output directories if they do not exist
if not os.path.exists('spectrograms'):
    os.makedirs('spectrograms')

for i, wav_file in enumerate(wav_files):
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)
    
    # Compute the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Create a plot for the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram of {wav_file}')
    plt.tight_layout()

    # Save the spectrogram as a PNG file
    plt.savefig(f'spectrograms/{i}.png')
    plt.close()
