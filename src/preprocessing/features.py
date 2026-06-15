import numpy as np
import librosa
from .config import N_MFCC, N_MELS


def extract_mfccs(segment, sr, n_mfcc=N_MFCC):
    """Return mean and std of each MFCC coefficient. Shape: (2 * n_mfcc,)."""
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])


def extract_mel_spectrogram(segment, sr, n_mels=N_MELS):
    """Return a dB-scaled mel spectrogram for segment. Shape: (n_mels, time_frames)."""
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)
