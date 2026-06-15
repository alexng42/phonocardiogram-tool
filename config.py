from pathlib import Path

DATA_DIR = (
    Path(__file__).resolve().parent
    / "classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
)
TRAINING_SUBSETS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
