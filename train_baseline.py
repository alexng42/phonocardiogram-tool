import sys, warnings, io
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              classification_report, confusion_matrix,
                              precision_recall_fscore_support)

from src.preprocessing.config import MIN_SEGMENTS

# ── Load ──────────────────────────────────────────────────────────────────────
npz        = np.load('features.npz', allow_pickle=True)
X          = npz['X_mfcc']
y          = npz['labels']
record_ids = npz['record_ids']

# ── Minimum-segment filter (applied before splitting) ─────────────────────────
unique_recs  = np.unique(record_ids)
seg_counts   = {r: int((record_ids == r).sum()) for r in unique_recs}

kept_recs    = np.array([r for r in unique_recs if seg_counts[r] >= MIN_SEGMENTS])
dropped_recs = np.array([r for r in unique_recs if seg_counts[r] <  MIN_SEGMENTS])

dropped_segs = sum(seg_counts[r] for r in dropped_recs)
kept_segs    = sum(seg_counts[r] for r in kept_recs)

keep_mask = np.isin(record_ids, kept_recs)
X_f  = X[keep_mask]
y_f  = y[keep_mask]
ids_f = record_ids[keep_mask]

rec_labels_f = np.array([y_f[ids_f == r][0] for r in kept_recs])

print("=== Minimum-Segment Filter (MIN_SEGMENTS={}) ===".format(MIN_SEGMENTS))
print(f"  Before : {len(unique_recs):4d} records | {len(y):6d} segments")
print(f"  Dropped: {len(dropped_recs):4d} records | {dropped_segs:6d} segments  "
      f"(all have < {MIN_SEGMENTS} segments)")
print(f"  Kept   : {len(kept_recs):4d} records | {kept_segs:6d} segments")

n_nor = (rec_labels_f ==  1).sum()
n_abn = (rec_labels_f == -1).sum()
print(f"  Class balance after filter: Normal {n_nor} ({n_nor/len(kept_recs)*100:.1f}%)  "
      f"Abnormal {n_abn} ({n_abn/len(kept_recs)*100:.1f}%)")

# ── Patient-level stratified 80/20 split on filtered set ─────────────────────
train_recs, test_recs = train_test_split(
    kept_recs, test_size=0.2, random_state=42, stratify=rec_labels_f
)
assert len(set(train_recs) & set(test_recs)) == 0, "Record leakage detected!"

train_mask = np.isin(ids_f, train_recs)
test_mask  = np.isin(ids_f, test_recs)
X_train, y_train = X_f[train_mask], y_f[train_mask]
X_test,  y_test  = X_f[test_mask],  y_f[test_mask]
test_ids = ids_f[test_mask]

print("\n=== Patient-level Train/Test Split ===")
for split, recs, ys in [('Train', train_recs, y_train), ('Test', test_recs, y_test)]:
    pct_n = (ys ==  1).mean() * 100
    pct_a = (ys == -1).mean() * 100
    print(f"  {split}: {len(recs):4d} records | {len(ys):6d} segments "
          f"| Normal {pct_n:.1f}%  Abnormal {pct_a:.1f}%")
print("  Record leakage check: PASSED")

# ── Model ─────────────────────────────────────────────────────────────────────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LinearSVC(class_weight='balanced', max_iter=3000, random_state=42)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ── Segment-level hold-out results ────────────────────────────────────────────
print("\n=== Segment-level (Hold-out Test Set) ===")
print(f"  Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred, labels=[-1, 1],
                             target_names=['Abnormal (-1)', 'Normal (+1)'], digits=4))

# ── Record-level aggregation ──────────────────────────────────────────────────
rec_true, rec_maj, rec_nsegs = [], [], []
for r in test_recs:
    mask = test_ids == r
    yt, yp = y_test[mask], y_pred[mask]
    rec_true.append(yt[0])
    rec_maj.append(-1 if (yp == -1).mean() > 0.5 else 1)
    rec_nsegs.append(mask.sum())

rec_true  = np.array(rec_true)
rec_maj   = np.array(rec_maj)
rec_nsegs = np.array(rec_nsegs)

def rec_report(label, yt, yp):
    n_nor = (yt ==  1).sum()
    n_abn = (yt == -1).sum()
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[-1, 1], average=None)
    bal = balanced_accuracy_score(yt, yp)
    print(f"  {label}  ({len(yt)} records: {n_nor} Normal, {n_abn} Abnormal)")
    print(f"    {'':20s}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'BalAcc':>7}")
    print(f"    {'Abnormal (-1)':<20}  {p[0]:>7.4f}  {r[0]:>7.4f}  {f[0]:>7.4f}")
    print(f"    {'Normal (+1)':<20}  {p[1]:>7.4f}  {r[1]:>7.4f}  {f[1]:>7.4f}  {bal:>7.4f}")

all_m     = np.ones(len(rec_true), dtype=bool)
typical_m = (rec_nsegs >= 10) & (rec_nsegs <= 29)

print("\n=== Majority-vote Record-level (Hold-out Test Set) ===")
for label, mask in [('All records', all_m), ('10-29 segments (typical)', typical_m)]:
    rec_report(label, rec_true[mask], rec_maj[mask])
    print()

# ── Stratified Group K-Fold CV ────────────────────────────────────────────────
print("=== Stratified Group 5-Fold CV (filtered dataset) ===")
cv   = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
y_cv = cross_val_predict(pipe, X_f, y_f, groups=ids_f, cv=cv, n_jobs=-1)

fold_acc, fold_bal, fold_f1_mac, fold_f1_wt = [], [], [], []
for _, tidx in cv.split(X_f, y_f, groups=ids_f):
    yt, yp = y_f[tidx], y_cv[tidx]
    fold_acc.append(accuracy_score(yt, yp))
    fold_bal.append(balanced_accuracy_score(yt, yp))
    fold_f1_mac.append(f1_score(yt, yp, average='macro'))
    fold_f1_wt.append(f1_score(yt, yp, average='weighted'))

def fmt(s): return f"{np.mean(s):.4f} +/- {np.std(s):.4f}  {np.round(s,4).tolist()}"
print(f"  {'Accuracy':<22}: {fmt(fold_acc)}")
print(f"  {'Balanced Accuracy':<22}: {fmt(fold_bal)}")
print(f"  {'F1 Macro':<22}: {fmt(fold_f1_mac)}")
print(f"  {'F1 Weighted':<22}: {fmt(fold_f1_wt)}")
print()
print("  Per-class report (aggregated over all CV folds):")
print(classification_report(y_f, y_cv, labels=[-1, 1],
                             target_names=['Abnormal (-1)', 'Normal (+1)'], digits=4))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
cm_hold = confusion_matrix(y_test, y_pred, labels=[-1, 1])
cm_cv   = confusion_matrix(y_f,    y_cv,   labels=[-1, 1])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm, title in [(axes[0], cm_hold, 'Hold-out (filtered)'),
                      (axes[1], cm_cv,   '5-fold CV (filtered)')]:
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred\nAbnormal', 'Pred\nNormal'])
    ax.set_yticklabels(['Act\nAbnormal', 'Act\nNormal'])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=13, color='white' if cm[i,j] > thresh else 'black')
    ax.set_title(f'LinearSVC — {title}')
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=130)
print("Saved: confusion_matrices.png")
