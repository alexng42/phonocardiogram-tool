import numpy as np
from scipy.signal import butter, filtfilt
from .config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER


def bandpass_filter(signal, sr, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, order=FILTER_ORDER):
    nyq = sr / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def normalize(signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val
