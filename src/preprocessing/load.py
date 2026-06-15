import librosa
from .config import SAMPLE_RATE


def load_pcg(filepath, target_sr=SAMPLE_RATE):
    """Load a PCG WAV file, resampling to target_sr if necessary."""
    signal, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return signal, sr


def get_label(wav_path):
    """Parse the paired .hea file and return 1 (normal) or -1 (abnormal).

    Returns None for files labelled as unlabelled/unknown so callers can skip them.
    Raises ValueError for any unrecognised label string.
    """
    hea_path = str(wav_path).replace(".wav", ".hea")
    with open(hea_path) as f:
        lines = f.read().strip().splitlines()

    last_line = lines[-1].strip()
    if "Normal" in last_line:
        return 1
    if "Abnormal" in last_line:
        return -1
    if "Unlabelled" in last_line or "Unlabeled" in last_line:
        return None
    raise ValueError(f"Unrecognised label in {hea_path!r}: {last_line!r}")
