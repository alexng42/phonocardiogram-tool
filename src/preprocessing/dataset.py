import os
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE
from .load import load_pcg, get_label
from .filters import bandpass_filter, normalize
from .segmentation import segment_heartbeats
from .features import extract_mfccs, extract_mel_spectrogram

_DATASET_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
)
TRAINING_SUBSETS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]


def preprocess_dataset(data_dir, sr=SAMPLE_RATE):
    """Process all WAV files in data_dir.

    Returns:
        X_mfcc  — (N, 2*n_mfcc) float array
        X_mel   — (N, n_mels, time_frames) float array
        labels  — (N,) int array of 1 (normal) / -1 (abnormal)
    """
    X_mfcc, X_mel, labels = [], [], []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".wav"):
            continue

        fpath = os.path.join(data_dir, fname)

        label = get_label(fpath)
        if label is None:
            continue  # skip unlabelled recordings

        signal, _ = load_pcg(fpath, sr)
        signal = bandpass_filter(signal, sr)
        signal = normalize(signal)

        segments = segment_heartbeats(signal, sr)

        for seg in segments:
            X_mfcc.append(extract_mfccs(seg, sr))
            X_mel.append(extract_mel_spectrogram(seg, sr))
            labels.append(label)

    return np.array(X_mfcc), np.array(X_mel), np.array(labels)


def preprocess_all_subsets(sr=SAMPLE_RATE):
    """Run preprocess_dataset over every training subset and concatenate results."""
    all_mfcc, all_mel, all_labels = [], [], []
    for subset in TRAINING_SUBSETS:
        X_mfcc, X_mel, y = preprocess_dataset(_DATASET_DIR / subset, sr)
        if len(y):
            all_mfcc.append(X_mfcc)
            all_mel.append(X_mel)
            all_labels.append(y)
    return np.concatenate(all_mfcc), np.concatenate(all_mel), np.concatenate(all_labels)
