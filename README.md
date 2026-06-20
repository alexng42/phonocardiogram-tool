# Phonocardiogram Classification

A proof-of-concept binary classifier for heart sound recordings. The pipeline
preprocesses raw PCG (phonocardiogram) audio, extracts MFCC features, and
trains a linear SVM to distinguish **Normal** from **Abnormal** heart sounds
using the PhysioNet/CinC 2016 Challenge dataset.

This is a course/portfolio project — not a diagnostic tool. It demonstrates a
complete end-to-end ML pipeline from raw audio to evaluated model, with
deliberate design choices documented below.

---

## Dataset

**Source:** PhysioNet/Computing in Cardiology Challenge 2016 —
*Classification of Heart Sound Recordings*
([https://physionet.org/content/challenge-2016/1.0.0/](https://physionet.org/content/challenge-2016/1.0.0/))

| Subset | Files |
|---|---|
| training-a | 409 |
| training-b | 490 |
| training-c | 31 |
| training-d | 55 |
| training-e | 2141 |
| training-f | 114 |
| **Total** | **3240** |

Class balance (labelled recordings): **2575 Normal (79.5%)**, **665 Abnormal (20.5%)**.

Recordings vary widely in duration and quality. All WAVs are natively 2000 Hz.
Labels are read from paired `.hea` files; unlabelled recordings are skipped.

The dataset is not included in this repository (see `.gitignore`). See
[Setup](#setup) for download instructions.

---

## Pipeline

```
WAV file
  └─ load (2000 Hz, mono)
      └─ bandpass filter (25–400 Hz, 4th-order Butterworth, zero-phase)
          └─ peak normalization
              └─ fixed-length windowing (2 s windows, 50% overlap)
                  ├─ MFCC extraction  → mean + std of 13 coefficients = 26 features/segment
                  └─ mel spectrogram  → 128 mel bins × 8 time frames/segment
```

All 3240 recordings produce **68,104 segments** in total. The bandpass range
(25–400 Hz) covers the diagnostically relevant heart sound frequencies while
rejecting low-frequency motion artifact and high-frequency noise above the
physiological range.

---

## Key Design Decisions

### Fixed-length windowing instead of beat segmentation

An early version used HeartPy to locate heartbeat peaks and extract
individual beat segments. This produced a systematic artifact: HeartPy
double-counted the S1 and S2 components of abnormal heart sounds as separate
peaks, yielding more segments per beat in abnormal recordings. The segmentation
count itself became a proxy for the label, which would constitute data leakage
if the model could exploit it indirectly through feature statistics.

Fixed-length overlapping windows (2 s, 50% overlap) are label-agnostic by
construction. Each window is long enough to span at least one full cardiac
cycle at normal resting heart rates (0.6–1.0 s per cycle), so MFCC statistics
aggregate over a representative portion of the signal without requiring peak
detection.

If beat-level segmentation is desired in future work, the
[Springer HMM segmenter](https://github.com/davidspringer/Springer-Segmentation-Code)
from the 2016 challenge is specifically designed for PCG and does not exhibit
the S1/S2 double-counting problem.

### Minimum-segment filter (`MIN_SEGMENTS = 10`)

Short recordings yield very few 2-second windows. Empirically, records with
fewer than 10 segments classified significantly worse than those with more,
and their small window count makes per-record MFCC statistics unreliable.
Excluding them improves model stability without discarding the bulk of the
data:

| | Records | Segments |
|---|---|---|
| Before filter | 3240 | 68,104 |
| Dropped (< 10 segments) | 671 | 4,865 |
| **Kept** | **2569** | **63,239** |

Keeping these short recordings in (pre-filter run): hold-out balanced accuracy
**0.7985**, 5-fold CV balanced accuracy **0.7920 ± 0.0131**.

After filtering: hold-out balanced accuracy **0.8247**, 5-fold CV balanced
accuracy **0.8173 ± 0.0035** — a consistent ~2.5 pp improvement.

### Label convention

`REFERENCE.csv` in each subset uses the PhysioNet 2016 convention:
`1 = Abnormal`, `-1 = Normal`. This pipeline uses the opposite: `1 = Normal`,
`-1 = Abnormal`. The relationship is exact: `ref_label == -pipeline_label` for
every record (verified at preprocessing time). This is documented in
`run_preprocessing.py` and printed at each preprocessing run.

---

## Model and Results

**Model:** `LinearSVC` with `class_weight='balanced'` and `StandardScaler`
(via a scikit-learn Pipeline). Features are the 26-dimensional MFCC vector
(mean and std of 13 coefficients) per segment.

**Evaluation methodology:**
- Patient-level stratified 80/20 split — all segments from a given recording
  appear in exactly one partition (no record leakage; verified by assertion).
- Train: **2055 records / 50,608 segments**; Test: **514 records / 12,631 segments**.
- Segment-level predictions are aggregated to record-level by majority vote
  (abnormal wins if > 50% of the record's segments are predicted abnormal).

### Segment-level (hold-out test set, filtered)

| | Precision | Recall | F1 |
|---|---|---|---|
| Abnormal | 0.5633 | 0.8690 | 0.6836 |
| Normal | 0.9481 | 0.7804 | 0.8561 |
| **Balanced Accuracy** | | | **0.8247** |

### Record-level majority vote (hold-out test set, filtered)

**All 514 test records** (407 Normal, 107 Abnormal):

| | Precision | Recall | F1 |
|---|---|---|---|
| Abnormal | 0.5625 | 0.9252 | 0.6996 |
| Normal | 0.9763 | 0.8108 | 0.8859 |
| **Balanced Accuracy** | | | **0.8680** |

**Records with 10–29 segments** (375 records; typical duration range):

| | Precision | Recall | F1 |
|---|---|---|---|
| Abnormal | 0.4904 | 0.9623 | 0.6497 |
| Normal | 0.9926 | 0.8354 | 0.9073 |
| **Balanced Accuracy** | | | **0.8988** |

### 5-fold stratified group cross-validation (filtered dataset)

Groups are by record ID — no record appears in both train and validation folds.

| Metric | Mean ± Std | Per-fold |
|---|---|---|
| Accuracy | 0.8111 ± 0.0080 | [0.8077, 0.8223, 0.7982, 0.8125, 0.8145] |
| Balanced Accuracy | 0.8173 ± 0.0035 | [0.8120, 0.8163, 0.8198, 0.8163, 0.8223] |
| F1 Macro | 0.7716 ± 0.0062 | [0.7672, 0.7801, 0.7626, 0.7723, 0.7760] |
| F1 Weighted | 0.8212 ± 0.0066 | [0.8181, 0.8304, 0.8106, 0.8224, 0.8246] |

---

## Limitations

**Segment-level vs. record-level.** The SVM is trained on individual 2-second
segments, but a clinical decision is made per recording. Majority vote is a
simple aggregation; it does not account for the temporal pattern of murmurs
within a recording.

**Short recording exclusion.** 671 recordings (20.7% of the labelled set) are
excluded by the minimum-segment filter. Performance on very short recordings
is not characterised, and those recordings may not be representative of the
kept set.

**Precision/recall asymmetry.** Abnormal recall is high (0.87–0.96 depending
on aggregation level), which is desirable for a screening scenario — few
abnormal recordings are missed. However, precision for the abnormal class is
low (0.49–0.56): roughly half of all abnormal predictions are false positives.
In a clinical deployment this would generate substantial referral burden on
specialists.

**MFCC ceiling.** Representing each segment by the mean and standard deviation
of its MFCC coefficients discards all temporal structure within the segment —
the timing relationship between S1 and S2, the duration of systole, and the
temporal pattern of a murmur are all collapsed into a single 26-dimensional
vector. This is likely the main bottleneck for further accuracy improvement
without architectural changes.

**Binary classification only.** The model distinguishes Normal from Abnormal
but does not identify specific pathologies (e.g., mitral regurgitation,
aortic stenosis, tricuspid abnormality). The dataset labels do not break
down abnormal sub-types.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` covers `librosa`, `numpy`, and `scipy`. You also need:

```bash
pip install scikit-learn matplotlib
```

Python 3.9+ recommended.

### Download the dataset

The dataset is not included in this repository. Download it from PhysioNet:

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2016/1.0.0/
```

Or use the PhysioNet web interface. Place (or symlink) the extracted directory
in the project root so the path is:

```
phonocardiogram-tool/
└── classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/
    ├── training-a/
    ├── training-b/
    ...
```

`config.py` constructs this path relative to the project root.

### Preprocessing

```bash
python run_preprocessing.py
```

Processes all 3240 WAV files (~5 min), writes `features.npz` (MFCC + mel
spectrogram arrays), `preprocessing_log.csv`, and `segments_vs_duration.png`.

### Training and evaluation

```bash
python train_baseline.py
```

Loads `features.npz`, applies the minimum-segment filter, runs a patient-level
80/20 split, fits the LinearSVC, and prints segment-level and record-level
metrics. Also runs 5-fold stratified group CV and saves `confusion_matrices.png`.

---

## Project Structure

```
phonocardiogram-tool/
├── config.py                    # dataset root path and subset list
├── run_preprocessing.py         # batch preprocessing → features.npz
├── train_baseline.py            # LinearSVC training, evaluation, confusion matrix plot
├── requirements.txt
└── src/
    ├── pipeline.py              # higher-level dataset preprocessing function
    └── preprocessing/
        ├── config.py            # all tuneable hyperparameters (SR, filter bounds, window params, MIN_SEGMENTS)
        ├── load.py              # WAV loader and .hea label parser (with convention note)
        ├── filters.py           # 4th-order Butterworth bandpass and peak normalization
        ├── segmentation.py      # fixed-length overlapping windowing
        ├── features.py          # MFCC (mean+std) and dB mel spectrogram extraction
        └── dataset.py           # per-subset preprocessing loop
```

Generated artifacts (not committed):

```
features.npz               # X_mfcc (68104, 26), X_mel (68104, 128, 8), labels, record_ids
preprocessing_log.csv      # per-file: label, duration, segment count
confusion_matrices.png     # hold-out and CV confusion matrices side by side
segments_vs_duration.png   # scatter of segment count vs duration, coloured by label
```

---

## Future Work

- **CNN on mel spectrograms.** The mel spectrogram arrays are already saved in
  `features.npz`. A 2D CNN trained directly on `X_mel` would retain the
  temporal structure within each 2-second window that MFCC statistics discard.

- **Sequential modelling over segments.** An LSTM or transformer operating over
  the ordered sequence of per-segment predictions could capture intra-recording
  temporal patterns (e.g., a murmur present only in systole).

- **Beat-level segmentation.** Replacing fixed windows with the
  [Springer HMM segmenter](https://github.com/davidspringer/Springer-Segmentation-Code)
  would produce one feature vector per cardiac cycle rather than per arbitrary
  window, giving the model access to beat-aligned S1/S2 structure.

- **Multi-condition classification.** Extending beyond binary labels to
  distinguish specific valve conditions would require a labelled dataset with
  sub-type annotations; the PhysioNet 2016 labels do not provide these.
