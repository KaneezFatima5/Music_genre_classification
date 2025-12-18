import os.path

from utils import *
from config import *
import time
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def create_spectrogram(train_music_path, test_music_path, write_path):
    start_time = time.time()
    convert_train_data(train_music_path, os.path.join(write_path, "train"))
    convert_test_data(test_music_path, os.path.join(write_path, "test"))
    end_time = time.time()
    print(f"Feature extraction took: {end_time - start_time} seconds")

def convert_test_data(test_music_path, write_path):
    convert_folder_data(test_music_path, write_path, "unknown")


def convert_train_data(train_music_path, write_path):
    for genre in os.listdir(train_music_path):
        convert_folder_data(os.path.join(train_music_path, genre), write_path, genre)

def convert_folder_data(music_folder_path, write_path, genre=""):
    print(f"Extracting information from {music_folder_path} music.")
    linear_write_path = os.path.join(write_path,"linear",genre)
    log_write_path = os.path.join(write_path,"log",genre)
    layl_write_path = os.path.join(write_path, "layl", genre)
    Path(linear_write_path).mkdir(parents=True, exist_ok=True)
    Path(log_write_path).mkdir(parents=True, exist_ok=True)
    Path(layl_write_path).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        for file in os.listdir(music_folder_path):
            executor.submit(
                convert_music_file_data, os.path.join(music_folder_path, file), linear_write_path, log_write_path, layl_write_path, file
            )

def convert_music_file_data(read_path, linear_write_path, log_write_path, layl_write_path, file_name):
    audio, sample_rate = librosa.load(read_path)
    stft_audio = librosa.stft(audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    y_audio = np.abs(stft_audio) ** 2
    plot_spectrogram(f"{linear_write_path}/{file_name}.png", y_audio, sample_rate, HOP_LENGTH)
    y_log_audio = librosa.power_to_db(y_audio)
    plot_spectrogram(f"{log_write_path}/{file_name}.png", y_log_audio, sample_rate, HOP_LENGTH)
    plot_spectrogram(f"{layl_write_path}/{file_name}.png", y_log_audio, sample_rate, HOP_LENGTH, "log")

def plot_spectrogram(write_path, y, sr, hop_length, y_axis = "linear"):
   plt.figure(figsize=(224/DPI, 224/DPI), dpi=DPI)
   librosa.display.specshow(y, sr = sr, hop_length = hop_length, x_axis="time", y_axis=y_axis)
   #plt.colorbar(format="%+2.f")
   plt.axis("off")
   plt.savefig(write_path, bbox_inches="tight", pad_inches=0)
   plt.close()

if __name__ == "__main__":
    create_spectrogram(TRAIN_MUSIC_PATH, TEST_MUSIC_PATH, SPECTROGRAM_PATH)