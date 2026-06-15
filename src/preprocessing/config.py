SAMPLE_RATE = 2000       # Hz — PhysioNet WAVs are natively 2000 Hz
BANDPASS_LOW = 25        # Hz
BANDPASS_HIGH = 400      # Hz
FILTER_ORDER = 4
N_MFCC = 13
N_MELS = 128
WINDOW_SEC = 2.0          # fixed window length for segmentation
WINDOW_OVERLAP = 0.5      # 50% overlap between consecutive windows
