import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report, confusion_matrix)

# ── Load ──────────────────────────────────────────────────────────────────────
npz        = np.load('features.npz', allow_pickle=True)
X          = npz['X_mfcc']          # (68104, 26)
y          = npz['labels']          # 1=Normal, -1=Abnormal
record_ids = npz['record_ids']

# ── Patient-level stratified 80/20 split ──────────────────────────────────────
unique_recs   = np.unique(record_ids)
rec_labels    = np.array([y[record_ids == r][0] for r in unique_recs])

train_recs, test_recs = train_test_split(
    unique_recs, test_size=0.2, random_state=42, stratify=rec_labels
)
assert len(set(train_recs) & set(test_recs)) == 0

train_mask = np.isin(record_ids, train_recs)
test_mask  = np.isin(record_ids, test_recs)
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print("=== Patient-level Train/Test Split ===")
for split, recs, ys in [('Train', train_recs, y_train), ('Test', test_recs, y_test)]:
    pct_n = (ys == 1).mean() * 100
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

# ── Hold-out results ──────────────────────────────────────────────────────────
print("\n=== Hold-out Test Set ===")
print(f"  Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
print()
print(classification_report(
    y_test, y_pred,
    labels=[-1, 1],
    target_names=['Abnormal (-1)', 'Normal (+1)'],
    digits=4,
))

cm_hold = confusion_matrix(y_test, y_pred, labels=[-1, 1])
print("  Confusion matrix (rows=actual, cols=predicted):")
print(f"  {'':20s}  {'Pred Abnormal':>14}  {'Pred Normal':>12}")
print(f"  {'Actual Abnormal':20s}  {cm_hold[0,0]:>14}  {cm_hold[0,1]:>12}")
print(f"  {'Actual Normal':20s}  {cm_hold[1,0]:>14}  {cm_hold[1,1]:>12}")

# ── Stratified Group K-Fold CV ────────────────────────────────────────────────
print("\n=== Stratified Group 5-Fold CV (grouped by record_id) ===")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_predict gives per-sample predictions across all folds
y_cv = cross_val_predict(pipe, X, y, groups=record_ids, cv=cv, n_jobs=-1)

# Compute per-fold metrics manually for mean ± std
fold_acc, fold_bal, fold_f1_macro, fold_f1_weighted = [], [], [], []
from sklearn.metrics import f1_score
for _, test_idx in cv.split(X, y, groups=record_ids):
    yt, yp = y[test_idx], y_cv[test_idx]
    fold_acc.append(accuracy_score(yt, yp))
    fold_bal.append(balanced_accuracy_score(yt, yp))
    fold_f1_macro.append(f1_score(yt, yp, average='macro'))
    fold_f1_weighted.append(f1_score(yt, yp, average='weighted'))

def fmt(scores):
    return f"{np.mean(scores):.4f} ± {np.std(scores):.4f}  {np.round(scores, 4).tolist()}"

print(f"  {'Accuracy':<22}: {fmt(fold_acc)}")
print(f"  {'Balanced Accuracy':<22}: {fmt(fold_bal)}")
print(f"  {'F1 Macro':<22}: {fmt(fold_f1_macro)}")
print(f"  {'F1 Weighted':<22}: {fmt(fold_f1_weighted)}")
print()
print("  Per-class report (aggregated over all CV folds):")
print(classification_report(
    y, y_cv,
    labels=[-1, 1],
    target_names=['Abnormal (-1)', 'Normal (+1)'],
    digits=4,
))

cm_cv = confusion_matrix(y, y_cv, labels=[-1, 1])

# ── Confusion matrix plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm, title in [
    (axes[0], cm_hold, 'Hold-out test set'),
    (axes[1], cm_cv,  '5-fold CV (all data)'),
]:
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred\nAbnormal', 'Pred\nNormal'])
    ax.set_yticklabels(['Act\nAbnormal', 'Act\nNormal'])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    fontsize=13, color='white' if cm[i, j] > thresh else 'black')
    ax.set_title(f'LinearSVC — {title}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=130)
print("Saved: confusion_matrices.png")
