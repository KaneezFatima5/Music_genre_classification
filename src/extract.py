from utils import *
from config import *
import time
import numpy as np
import librosa
import pandas as pd
# import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract(
    train_music_path, test_music_path, train_spreadsheet_path, test_spreadsheet_path
):
    start_time = time.time()
    write_spreadsheet(extract_train_data(train_music_path), train_spreadsheet_path)
    write_spreadsheet(extract_test_data(test_music_path), test_spreadsheet_path)
    end_time = time.time()
    # print(f"Feature extraction took: {end_time - start_time} seconds")


def extract_test_data(test_music_path):
    return extract_folder_data(test_music_path, "unknown")


def extract_train_data(train_music_path):
    dataframes = []
    for genre in os.listdir(train_music_path):
        dataframes.append(
            extract_folder_data(os.path.join(train_music_path, genre), genre)
        )
    return pd.concat(dataframes, ignore_index=True)


def extract_folder_data(genre_path, genre):
    results = []
    # print(f"Extracting information from {genre} music.")
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                extract_music_file_data, os.path.join(genre_path, file), file, genre
            )
            for file in os.listdir(genre_path)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing a file: {e}")


    columns = [
        INDEX_COLUMN,
        "Tempo",
        "BeatCount",
        "HarmonicMean",
        # "HarmonicMedian",
        # "HarmonicMin",
        "HarmonicMax",
        "PercussiveMean",
        "PercussiveMedian",
        "PercussiveMin",
        # "PercussiveMax",
        # "rms_average",
        # "rms_std",
        # "rms_max",
        "dynamic_range"
    ]
    # Chroma features (12) 
    columns += [f"chroma_mean_{i}" for i in range(12)] 
    columns += [f"chroma_std_{i}" for i in range(12)]
    # MFCC features (20) 
    columns += [f"mfcc_{i}" for i in range(13)] 
    # Spectral features 
    columns+=[
        # "zero_crossing_rate_average",
        # "zero_crossing_rate_std",
        # "chroma_average",
        # "chroma_std",
        # "MFCC_average"
        # "MFCC_std",
        # "spectral_centroid_average",
        "spectral_centroid_std",
        # "spectral_centroid_max",
        "spectral_bandwidth_average",
        # "spectral_bandwidth_std",
        # "spectral_bandwidth_max",
        # "spectral_rolloff_average",
        "spectral_rolloff_std",
        "spectral_rolloff_max"
    ]
    # Spectral contrast (7 bands by default in librosa) 
    columns += [f"spectral_contrast_mean_{i}" for i in range(7)] 
    columns += [f"spectral_contrast_std_{i}" for i in range(7)] 
    columns += [f"spectral_contrast_max_{i}" for i in range(7)] 
    # Spectral flatness 
    columns += [

        "spectral_flatness_average",
        # "spectral_flatness_std",
        "spectral_flatness_max",
        "onset_strength_average",
        # "onset_strength_std",
        "onset_strength_max",
        # "low_energy_ratio",
        "mid_energy_ratio",
        "high_energy_ratio"
    ]
    # Mel spectrogram (128 by default) 
    # columns += [f"mel_mean_{i}" for i in range(128)] 
    # columns += [f"mel_std_{i}" for i in range(128)] 
    # Spectral flux 
    columns += [
        "mel_mean",
        "mel_std",
        "flux_mean",
        "flux_std",
        CLASS_COLUMN,
    ]
    return pd.DataFrame(results, columns=columns)


def get_short_time_fourier_transform(y):
    return librosa.stft(y=y)  # 2D array for frequency bins and time frames


def extract_music_file_data(file_path, file_name, genre):
    collected_data = [file_name]
    y, sr = preprocess_music_data(file_path)
    stft = get_short_time_fourier_transform(y)
    collected_data.extend(get_tempo_and_beats(y, sr))
    collected_data.extend(get_harmonic_and_percussive_components(y, stft))
    collected_data.append(rms_features(y))
    chroma_mean, chroma_std = chroma_features(y, sr)
    collected_data.extend(chroma_mean)
    collected_data.extend(chroma_std)
    mfcc_mean = mfcc_features(y, sr)
    collected_data.extend(mfcc_mean)
    collected_data.append(spectral_centroid_features(y, sr))
    collected_data.append(spectral_bandwidth_features(y, sr))
    collected_data.extend(spectral_rolloff_features(y, sr))
    contrast_mean, contrast_std, contrast_max = spectral_contrast_features(y, sr)
    collected_data.extend(contrast_mean)
    collected_data.extend(contrast_std)
    collected_data.extend(contrast_max)
    collected_data.extend(spectral_flatness_features(y, sr))
    collected_data.extend(onset_strength_features(y, sr))
    collected_data.extend(freq_energy_ratios(y, sr, stft))
    # mel_mean, mel_std = mel_spectogram(y, sr)
    collected_data.extend(mel_spectogram(y, sr))
    # collected_data.extend(mel_std)
    collected_data.extend(spectral_flux(y, sr))
    collected_data.append(genre)
    return collected_data


def preprocess_music_data(file_path):
    y, sr = librosa.load(file_path, sr=DESIRED_SR)
    return y, sr


def get_tempo_and_beats(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    beat_count = len(beat_frames)
    # print(f"find tempo{tempo}")
    return tempo[0], beat_count


def get_harmonic_and_percussive_components(y, stft):  # input: amplitude (signal array)
    # harmonic-percussive source separation
    harmonic, percussive = librosa.decompose.hpss(stft, margin=HPSS_MARGIN)
    # get time series audio arrays
    y_harmonic = librosa.istft(harmonic)
    y_percussive = librosa.istft(percussive)
    return extract_features_hp_components(y_harmonic, y_percussive)


def extract_features_hp_components(y_harmonic, y_percussive):
    # harmonic features
    harmonic_mean = np.mean(np.abs(y_harmonic))
    # harmonic_median = np.median(np.abs(y_harmonic))
    # harmonic_min = np.min(y_harmonic)
    harmonic_max = np.max(y_harmonic)

    # percussive features
    percussive_mean = np.mean(np.abs(y_percussive))
    percussive_median = np.median(np.abs(y_percussive))
    percussive_min = np.min(y_percussive)
    # percussive_max = np.max(y_percussive)

    return (
        harmonic_mean,
        # harmonic_median,
        # harmonic_min,
        harmonic_max,
        percussive_mean,
        percussive_median,
        percussive_min,
        # percussive_max,
    )


def rms_features(y):
    """Root mean square (RMS) energy, Dynamic range and zero Crossing rate"""
    rms_value = librosa.feature.rms(y=y)
    # rms_mean = np.mean(rms_value)
    # rms_std = np.std(rms_value)
    rms_max = np.max(rms_value)
    rms_min = np.min(rms_value)

    dynamic_range = rms_max - rms_min
    # zero crossing rate (ZCR) feature
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    # zcr_mean = np.mean(zcr)
    # zcr_std = np.std(zcr)
    return dynamic_range


def chroma_features(y, sr):  # amplitude and sample rate
    """Get Chroma Features"""
    chroma_val = librosa.feature.chroma_stft(
        y=y, sr=sr
    )  # gives the strength of 12 notes at diff points
    chroma_mean = np.mean(chroma_val, axis=1)
    chroma_std = np.std(chroma_val, axis=1)
    # print(chroma_mean)
    return chroma_mean, chroma_std


def mfcc_features(y, sr):
    """Get Mel-Frequency Cepstral Coefficients (MFCC)"""
    mfcc_val = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc_val, axis=1)
    # mfcc_std = np.std(mfcc_val)
    # print(f"mfcc{mfcc_mean}")
    return mfcc_mean


def spectral_centroid_features(y, sr):
    """Get Spectral Shape Features"""
    centroid_val = librosa.feature.spectral_centroid(y=y, sr=sr)
    frame_times = librosa.frames_to_time(np.arange(centroid_val.shape[1]), sr=sr)
    centroid = centroid_val[0][frame_times > 1.5]
    # if centroid.size == 0:
    #     # centroid_mean = centroid_std = 0
    # else:
    # centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)
    # centroid_max = np.max(centroid)
    return centroid_std


#
def spectral_bandwidth_features(y, sr):
    """Get Spectral BandWidth Features"""
    bandwidth_val = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    frame_times = librosa.frames_to_time(np.arange(bandwidth_val.shape[1]), sr=sr)
    bandwidth = bandwidth_val[0][frame_times > 1.5]
    bandwidth_mean = np.mean(bandwidth)
    # bandwidth_std = np.std(bandwidth)
    # bandwidth_max = np.max(bandwidth)
    return bandwidth_mean


def spectral_rolloff_features(y, sr):
    """Get Spectral RollOff Features"""
    rolloff_val = librosa.feature.spectral_rolloff(y=y, sr=sr)
    frame_times = librosa.frames_to_time(np.arange(rolloff_val.shape[1]), sr=sr)
    rolloff = rolloff_val[0][frame_times > 1.5]
    # rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    rolloff_max = np.max(rolloff)
    return rolloff_std, rolloff_max


def spectral_contrast_features(y, sr):
    """Get Spectral Contrast Features"""
    contrast_val = librosa.feature.spectral_contrast(y=y, sr=sr)
    # frame_times = librosa.frames_to_time(np.arange(contrast_val.shape[1]), sr=sr)
    # contrast = contrast_val[0][frame_times > 1.5]
    contrast_mean = np.mean(contrast_val, axis=1)
    contrast_std = np.std(contrast_val, axis=1)
    contrast_max = np.max(contrast_val, axis=1)
    # print(f"{contrast_max}")
    # print(f"{contrast_std}")
    # print(f"{contrast_mean}")

    return contrast_mean, contrast_std, contrast_max


def spectral_flatness_features(y, sr):
    """Spectral Flatness Features"""
    flatness_val = librosa.feature.spectral_flatness(y=y)
    frame_times = librosa.frames_to_time(np.arange(flatness_val.shape[1]), sr=sr)
    flatness = flatness_val[0][frame_times > 1.5]
    flatness_mean = np.mean(flatness)
    # flatness_std = np.std(flatness)
    flatness_max = np.max(flatness)
    return flatness_mean, flatness_max


def onset_strength_features(y, sr):
    """Captures the energy and intensity of attacks in the music"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # onset_strength_mean = np.mean(onset_env)
    onset_std = np.std(onset_env)
    onset_max = np.max(onset_env)
    return onset_std, onset_max


def freq_energy_ratios(y, sr, stft):
    """Get the Frequency Energy Ratios"""
    spectrogram = np.abs(stft)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Frequency Bands
    low_idx = np.where(freq_bins < 500)[0]
    mid_idx = np.where((freq_bins >= 500) & (freq_bins < 2000))[0]
    high_idx = np.where(freq_bins >= 2000)[0]

    # sum energy across bands
    low_energy = np.sum(spectrogram[low_idx, :], axis=0)
    mid_energy = np.sum(spectrogram[mid_idx, :], axis=0)
    high_energy = np.sum(spectrogram[high_idx, :], axis=0)

    total_energy = low_energy + mid_energy + high_energy

    valid = total_energy > 0
    low_ratio = np.zeros_like(total_energy)
    mid_ratio = np.zeros_like(total_energy)
    high_ratio = np.zeros_like(total_energy)

    low_ratio[valid] = low_energy[valid] / total_energy[valid]
    mid_ratio[valid] = mid_energy[valid] / total_energy[valid]
    high_ratio[valid] = high_energy[valid] / total_energy[valid]

    # low_energy_ratio = np.mean(low_ratio)
    mid_energy_ratio = np.mean(mid_ratio)
    high_energy_ratio = np.mean(high_ratio)

    return mid_energy_ratio, high_energy_ratio

def mel_spectogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel)
    mel_std = np.std(mel)
    # print(f"mel{mel_mean}")
    # print(f"mel{mel_std}")
    return mel_mean, mel_std

def spectral_flux(y, sr):
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    flux_mean = np.mean(flux)
    flux_std = np.std(flux)
    return flux_mean, flux_std

# def plot_data(y, sr):
#     """plotting Amplitude Vs Time"""
#     plt.figure(figsize=(20, 4))
#     librosa.display.waveshow(y, sr=sr, alpha=0.3, label="Rock", color="#DC143C")
#     plt.title("Waveforms of our Rock Song")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude")
#     plt.tight_layout()
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    extract(TRAIN_MUSIC_PATH, TEST_MUSIC_PATH, TRAIN_PATH, TEST_PATH)