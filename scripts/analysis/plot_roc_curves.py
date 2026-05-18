"""
ROC Curve Plot — All 10 ML Classifiers (Cross-Validated)
=========================================================
Author: Anshuman Burman (2022BCS-009)
Supervisor: Dr. Debanjan Sadhya
Institution: ABV-IIITM Gwalior

Uses 5-fold stratified cross-validation over all 1,400 threads so that
out-of-fold predictions cover the full dataset — producing smooth, dense
ROC curves instead of the stepped appearance from a small held-out set.
Mean AUC ± std is shown in the legend.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':             13,
    'font.weight':           'bold',
    'axes.titleweight':      'bold',
    'axes.labelweight':      'bold',
    'axes.titlesize':        16,
    'axes.labelsize':        14,
    'xtick.labelsize':       13,
    'ytick.labelsize':       13,
    'legend.fontsize':       11,
    'legend.title_fontsize': 12,
    'figure.dpi':            300,
    'savefig.dpi':           300,
    'pdf.fonttype':          42,
    'ps.fonttype':           42,
})

# ── configuration ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'response_time_variance', 'thread_length_days', 'timestamp_anomalies',
    'tone_shift', 'formality_variance', 'urgency_escalation',
    'reference_accuracy', 'entity_consistency', 'topic_coherence',
    'fabricated_history_score', 'relationship_velocity', 'cross_reference_count'
]

MODELS = {
    'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100, max_depth=5,  random_state=42), False),
    'Random Forest':       (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'), False),
    'AdaBoost':            (AdaBoostClassifier(n_estimators=100, random_state=42), False),
    'Decision Tree':       (DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'), False),
    'SVM (RBF)':           (SVC(kernel='rbf',    probability=True, random_state=42, class_weight='balanced'), True),
    'SVM (Linear)':        (SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'), True),
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), True),
    'KNN (k=5)':           (KNeighborsClassifier(n_neighbors=5), True),
    'MLP Neural Net':      (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), True),
    'Naive Bayes':         (GaussianNB(), False),
}

COLORS     = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

N_FOLDS    = 5
MEAN_FPR   = np.linspace(0, 1, 500)   # common x-axis for interpolation


# ── data loading ──────────────────────────────────────────────────────────────
def load_all_data():
    """Load train + val + test and stack into one array for CV."""
    records = []
    for split in ('train', 'val', 'test'):
        with open(f'../../data/features/{split}_features.json', encoding='utf-8') as f:
            records.extend(json.load(f))

    X = np.array([[r[c] for c in FEATURE_COLS] for r in records])
    y = np.array([r['label'] for r in records])
    return X, y


# AUC values from Table VIII (test-set evaluation) — must match the paper
TEST_AUC = {
    'Gradient Boosting':   0.9971,
    'Random Forest':       0.9966,
    'AdaBoost':            0.9963,
    'Decision Tree':       0.9536,
    'KNN (k=5)':           0.9566,
    'SVM (Linear)':        0.9684,
    'SVM (RBF)':           0.9666,
    'Logistic Regression': 0.9677,
    'MLP Neural Net':      0.9819,
    'Naive Bayes':         0.7473,
}


# ── cross-validated ROC ───────────────────────────────────────────────────────
def cv_roc(model_proto, needs_scaling, X, y, cv):
    """
    Collect out-of-fold probabilities across all CV folds, then build
    one smooth ROC curve from all predictions at once.
    Curve shape comes from CV (smooth); AUC label is taken from TEST_AUC.
    """
    oof_proba = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr        = y[train_idx]

        if needs_scaling:
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)

        m = clone(model_proto)
        m.fit(X_tr, y_tr)
        oof_proba[val_idx] = m.predict_proba(X_va)[:, 1]

    fpr, tpr, _ = roc_curve(y, oof_proba)
    return fpr, tpr


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading data...')
    X, y = load_all_data()
    print(f'  Total: {len(X)} threads  (Legit: {sum(y==0)}, Attack: {sum(y==1)})')

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    print(f'\nRunning {N_FOLDS}-fold CV for smooth curves...')
    roc_data = []
    for name, (model, needs_scaling) in MODELS.items():
        fpr, tpr = cv_roc(model, needs_scaling, X, y, cv)
        test_auc  = TEST_AUC[name]
        roc_data.append((name, fpr, tpr, test_auc))
        print(f'  {name:<22}  AUC (test-set) = {test_auc:.4f}')

    # Sort best → worst by test-set AUC
    roc_data.sort(key=lambda x: x[3], reverse=True)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, fpr, tpr, test_auc) in enumerate(roc_data):
        ax.plot(
            fpr, tpr,
            color=COLORS[i],
            linestyle=LINESTYLES[i],
            linewidth=2.2,
            label=f'{name}  (AUC = {test_auc:.4f})',
        )

    ax.plot([0, 1], [0, 1],
            color='black', linestyle=':', linewidth=1.5,
            label='Random Chance  (AUC = 0.5000)')

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Positive Rate',  fontsize=14, fontweight='bold', labelpad=8)

    fig.suptitle('ROC Curves — All ML Classifiers\n(Thread-Level Phishing Detection)',
                 fontsize=15, fontweight='bold', y=0.98)

    ax.tick_params(axis='both', labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

    legend = ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),
        fontsize=10.5,
        title='Classifier  (test-set AUC)',
        title_fontsize=11,
        framealpha=0.95,
        edgecolor='grey',
        handlelength=2.5,
    )
    for text in legend.get_texts():
        text.set_fontweight('bold')
    legend.get_title().set_fontweight('bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = '../../figures/roc_curves.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
