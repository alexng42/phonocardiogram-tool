import sys, os, csv, warnings, time, math
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing.load import load_pcg, get_label
from src.preprocessing.filters import bandpass_filter, normalize
from src.preprocessing.segmentation import segment_heartbeats
from src.preprocessing.features import extract_mfccs, extract_mel_spectrogram
from config import DATA_DIR, TRAINING_SUBSETS

t0 = time.time()

log_rows = []
all_mfcc, all_mel, all_labels, all_ids = [], [], [], []
failures = []

for subset in TRAINING_SUBSETS:
    subset_dir = DATA_DIR / subset
    wavfiles = sorted(f for f in os.listdir(subset_dir) if f.endswith('.wav'))
    print(f"Processing {subset}: {len(wavfiles)} files ...", flush=True)

    for fname in wavfiles:
        fpath = os.path.join(subset_dir, fname)
        record_id = fname.replace('.wav', '')

        try:
            label = get_label(fpath)
            if label is None:
                failures.append((subset, fname, 'unlabelled'))
                continue

            signal, sr = load_pcg(fpath)
            duration = len(signal) / sr
            signal = bandpass_filter(signal, sr)
            signal = normalize(signal)
            segments = segment_heartbeats(signal, sr)

            for seg in segments:
                all_mfcc.append(extract_mfccs(seg, sr))
                all_mel.append(extract_mel_spectrogram(seg, sr))
                all_labels.append(label)
                all_ids.append(record_id)

            log_rows.append({
                'file': fname, 'subset': subset, 'label': label,
                'duration_s': round(duration, 2), 'segments': len(segments),
            })

        except Exception as e:
            failures.append((subset, fname, str(e)))

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min — {len(log_rows)} files processed, {len(failures)} failures", flush=True)

# --- Save log CSV ---
with open('preprocessing_log.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'subset', 'label', 'duration_s', 'segments'])
    writer.writeheader()
    writer.writerows(log_rows)

# --- Label comparison vs REFERENCE.csv ---
# REFERENCE.csv convention: 1 = Abnormal, -1 = Normal  (PhysioNet 2016)
# Pipeline convention:      1 = Normal,   -1 = Abnormal (intuitive)
# Relationship: ref_label == -pipeline_label for every record
ref_counts = {}
for subset in TRAINING_SUBSETS:
    ref_path = DATA_DIR / subset / 'REFERENCE.csv'
    with open(ref_path) as f:
        for line in f:
            record, lbl = line.strip().split(',')
            ref_counts[record] = int(lbl)

ref_normal   = sum(1 for v in ref_counts.values() if v == -1)
ref_abnormal = sum(1 for v in ref_counts.values() if v ==  1)

pipe_labels  = {row['file'].replace('.wav', ''): row['label'] for row in log_rows}
pipe_normal   = sum(1 for v in pipe_labels.values() if v ==  1)
pipe_abnormal = sum(1 for v in pipe_labels.values() if v == -1)

print("\n=== Label Convention ===")
print("  REFERENCE.csv : 1 = Abnormal,  -1 = Normal  (PhysioNet 2016 convention)")
print("  Pipeline      : 1 = Normal,    -1 = Abnormal (intuitive convention)")
print("  Relationship  : ref_label == -pipeline_label")

print("\n=== Label Counts vs REFERENCE.csv ===")
print(f"  {'Source':<20} {'Normal':>10} {'Abnormal':>10} {'Total':>8}")
print(f"  {'REFERENCE.csv':<20} {ref_normal:>10} {ref_abnormal:>10} {ref_normal+ref_abnormal:>8}")
print(f"  {'Pipeline':<20} {pipe_normal:>10} {pipe_abnormal:>10} {pipe_normal+pipe_abnormal:>8}")
match = (ref_normal == pipe_normal and ref_abnormal == pipe_abnormal)
print(f"  Counts match: {match}")

mismatches = [
    (rec, pl, ref_counts[rec])
    for rec, pl in pipe_labels.items()
    if rec in ref_counts and ref_counts[rec] != -pl
]
if mismatches:
    print(f"\n  Unexpected per-record mismatches (beyond convention inversion): {len(mismatches)}")
    for rec, pl, rl in mismatches[:10]:
        print(f"    {rec}: pipeline={pl}, reference={rl}")
else:
    print("  Per-record check: all labels consistent with convention inversion.")

# --- Segment count vs duration plot ---
durations   = [row['duration_s'] for row in log_rows]
seg_counts  = [row['segments']   for row in log_rows]
labels_plot = [row['label']      for row in log_rows]

fig, ax = plt.subplots(figsize=(10, 6))
for lbl, color, name in [(1, 'steelblue', 'Normal'), (-1, 'tomato', 'Abnormal')]:
    d = [dur for dur, lb in zip(durations, labels_plot) if lb == lbl]
    s = [seg for seg, lb in zip(seg_counts, labels_plot) if lb == lbl]
    ax.scatter(d, s, c=color, alpha=0.35, s=8, label=f'{name} (n={len(d)})')

t_dur = sorted(set(durations))
t_seg = [math.floor((d - 2.0) / 1.0) + 1 for d in t_dur]
ax.plot(t_dur, t_seg, 'k--', linewidth=1, alpha=0.7, label='Expected (2s window, 50% overlap)')

ax.set_xlabel('Duration (s)')
ax.set_ylabel('Segment count')
ax.set_title('Segment count vs. duration — all training subsets')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('segments_vs_duration.png', dpi=130)
print("\nPlot saved: segments_vs_duration.png")

# --- Save .npz ---
X_mfcc = np.array(all_mfcc)
X_mel  = np.array(all_mel)
y      = np.array(all_labels)
ids    = np.array(all_ids)

np.savez_compressed('features.npz', X_mfcc=X_mfcc, X_mel=X_mel, labels=y, record_ids=ids)

print("\n=== Feature Arrays ===")
print(f"  X_mfcc     : {X_mfcc.shape}")
print(f"  X_mel      : {X_mel.shape}")
print(f"  labels     : {y.shape}  unique={sorted(set(y.tolist()))}")
print(f"  record_ids : {ids.shape}  unique records={len(set(ids.tolist()))}")

print(f"\n=== Failures ({len(failures)}) ===")
if failures:
    for sub, fname, reason in failures[:20]:
        print(f"  {sub}/{fname}: {reason}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more (see preprocessing_log.csv)")
else:
    print("  None")

print("\nSaved: preprocessing_log.csv, features.npz, segments_vs_duration.png")
