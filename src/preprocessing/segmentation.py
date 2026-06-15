from .config import WINDOW_SEC, WINDOW_OVERLAP


def segment_heartbeats(signal, sr, window_sec=WINDOW_SEC, overlap=WINDOW_OVERLAP):
    """Slice signal into fixed-length overlapping windows.

    Args:
        window_sec: window duration in seconds.
        overlap:    fractional overlap between consecutive windows (0–1).

    Returns a list of 1-D numpy arrays, each exactly sr * window_sec samples.
    """
    window_len = int(sr * window_sec)
    hop = int(window_len * (1 - overlap))
    return [
        signal[start : start + window_len]
        for start in range(0, len(signal) - window_len + 1, hop)
    ]
