"""
ROC Curve Plot — All 10 ML Classifiers
=======================================
Author: Anshuman Burman (2022BCS-009)
Supervisor: Dr. Debanjan Sadhya
Institution: ABV-IIITM Gwalior

Trains all 10 classifiers on the 12 hand-crafted thread-level features,
computes per-model ROC curves on the held-out test set, and saves a
single publication-ready figure suitable for two-column LaTeX.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':           13,
    'font.weight':         'bold',
    'axes.titleweight':    'bold',
    'axes.labelweight':    'bold',
    'axes.titlesize':      16,
    'axes.labelsize':      14,
    'xtick.labelsize':     13,
    'ytick.labelsize':     13,
    'legend.fontsize':     11,
    'legend.title_fontsize': 12,
    'figure.dpi':          300,
    'savefig.dpi':         300,
    'pdf.fonttype':        42,
    'ps.fonttype':         42,
})

# ── configuration ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'response_time_variance', 'thread_length_days', 'timestamp_anomalies',
    'tone_shift', 'formality_variance', 'urgency_escalation',
    'reference_accuracy', 'entity_consistency', 'topic_coherence',
    'fabricated_history_score', 'relationship_velocity', 'cross_reference_count'
]

MODELS = {
    'Gradient Boosting':  (GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42), False),
    'Random Forest':      (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'), False),
    'AdaBoost':           (AdaBoostClassifier(n_estimators=100, random_state=42), False),
    'Decision Tree':      (DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'), False),
    'SVM (RBF)':          (SVC(kernel='rbf',    probability=True, random_state=42, class_weight='balanced'), True),
    'SVM (Linear)':       (SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'), True),
    'Logistic Regression':(LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), True),
    'KNN (k=5)':          (KNeighborsClassifier(n_neighbors=5), True),
    'MLP Neural Net':     (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), True),
    'Naive Bayes':        (GaussianNB(), False),
}

# 10 visually distinct colours (tab10)
COLORS = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

# line styles to help distinguish when printed in B&W
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']


# ── data loading ──────────────────────────────────────────────────────────────
def load_data():
    with open('../../data/features/train_features.json', encoding='utf-8') as f:
        train = json.load(f)
    with open('../../data/features/test_features.json', encoding='utf-8') as f:
        test = json.load(f)

    X_train = np.array([[r[c] for c in FEATURE_COLS] for r in train])
    y_train = np.array([r['label'] for r in train])
    X_test  = np.array([[r[c] for c in FEATURE_COLS] for r in test])
    y_test  = np.array([r['label'] for r in test])
    return X_train, y_train, X_test, y_test


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading data...')
    X_train, y_train, X_test, y_test = load_data()

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f'  Train: {len(X_train)}  Test: {len(X_test)}')

    # ── train & collect ROC data ───────────────────────────────────────────────
    roc_data = []
    for name, (model, needs_scaling) in MODELS.items():
        Xtr = X_train_sc if needs_scaling else X_train
        Xte = X_test_sc  if needs_scaling else X_test
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        roc_data.append((name, fpr, tpr, roc_auc))
        print(f'  {name:<22}  AUC = {roc_auc:.4f}')

    # Sort descending by AUC so the legend reads best-to-worst
    roc_data.sort(key=lambda x: x[3], reverse=True)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, fpr, tpr, roc_auc) in enumerate(roc_data):
        ax.plot(
            fpr, tpr,
            color=COLORS[i],
            linestyle=LINESTYLES[i],
            linewidth=2.2,
            label=f'{name}  (AUC = {roc_auc:.3f})',
        )

    # Random-chance diagonal
    ax.plot([0, 1], [0, 1],
            color='black', linestyle=':', linewidth=1.5, label='Random Chance (AUC = 0.500)')

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

    # Legend outside the plot so it never overlaps curves
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
