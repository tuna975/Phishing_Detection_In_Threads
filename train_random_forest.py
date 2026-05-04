"""
Baseline Model: Random Forest Classifier for Thread-Level Phishing Detection
=============================================================================
Author: Anshuman Burman (2022BCS-009)
Supervisor: Dr. Debanjan Sadhya
Institution: ABV-IIITM Gwalior

This script trains a Random Forest classifier on 12 handcrafted thread-level
features for phishing detection. Serves as the baseline model for comparison
with the thread-aware BERT model.

Features (12):
  Temporal (3):    response_time_variance, thread_length_days, timestamp_anomalies
  Linguistic (3):  tone_shift, formality_variance, urgency_escalation
  Coherence (3):   reference_accuracy, entity_consistency, topic_coherence
  Context (3):     fabricated_history_score, relationship_velocity, cross_reference_count

Citations:
  - Verma et al. (2012), Das et al. (2019), Hasan et al. (2025)
  - Pedregosa et al. (2011) - scikit-learn
"""

import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_COLS = [
    'response_time_variance', 'thread_length_days', 'timestamp_anomalies',
    'tone_shift', 'formality_variance', 'urgency_escalation',
    'reference_accuracy', 'entity_consistency', 'topic_coherence',
    'fabricated_history_score', 'relationship_velocity', 'cross_reference_count'
]

ATTACK_TYPES = [
    'legitimate', 'thread_hijacking', 'relationship_exploitation',
    'context_manipulation', 'urgency_escalation'
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_features(filepath):
    """Load feature JSON file and return X, y arrays"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    X = np.array([[t[f] for f in FEATURE_COLS] for t in data])
    y = np.array([t['label'] for t in data])
    
    return X, y, data


# ============================================================================
# TRAINING
# ============================================================================

def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance (105 legit vs 269 attack)
    )
    rf.fit(X_train, y_train)
    return rf


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(rf, X_test, y_test, test_data):
    """Comprehensive model evaluation"""
    
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    
    # Overall metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n  Accuracy:   {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}")
    
    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Attack']))
    
    # Confusion matrix
    print("--- Confusion Matrix ---")
    print(f"                  Predicted")
    print(f"               Legit  Attack")
    print(f"  Actual Legit  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Actual Attack {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Feature importance
    print("\n--- Feature Importance (Ranked) ---")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices):
        bar = "█" * int(importances[idx] * 50)
        print(f"  {rank+1:2d}. {FEATURE_COLS[idx]:30s} {importances[idx]:.4f}  {bar}")
    
    # Per attack type
    print("\n--- Performance by Attack Type ---")
    for atype in ATTACK_TYPES:
        mask = [i for i, t in enumerate(test_data) if t['attack_type'] == atype]
        if mask:
            correct = int(sum(y_pred[i] == y_test[i] for i in mask))
            total = len(mask)
            print(f"  {atype:30s}: {correct}/{total} correct ({correct/total*100:.1f}%)")
    
    # Misclassified threads
    misclassified = [
        {"thread_id": test_data[i]['thread_id'], "attack_type": test_data[i]['attack_type'],
         "true": int(y_test[i]), "predicted": int(y_pred[i]), 
         "prob": round(float(y_proba[i]), 4)}
        for i in range(len(y_test)) if y_pred[i] != y_test[i]
    ]
    
    if misclassified:
        print(f"\n--- Misclassified Threads ({len(misclassified)}) ---")
        for m in misclassified:
            label = "Attack" if m['true'] == 1 else "Legit"
            pred = "Attack" if m['predicted'] == 1 else "Legit"
            print(f"  {m['thread_id']:25s} | Type: {m['attack_type']:25s} | True: {label} → Pred: {pred} (prob: {m['prob']})")
    else:
        print("\n  No misclassifications! (Perfect on test set)")
    
    return y_pred, y_proba


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(rf, y_test, y_pred, y_proba, test_data):
    """Save model and results to files"""
    
    # Save model
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print("\n✅ Model saved: random_forest_model.pkl")
    
    # Save detailed results
    cm = confusion_matrix(y_test, y_pred)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    results = {
        "model": "Random Forest Classifier (Baseline)",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42
        },
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4)
        },
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1])
        },
        "feature_importance": [
            {"rank": r+1, "feature": FEATURE_COLS[idx], 
             "importance": round(float(importances[idx]), 4)}
            for r, idx in enumerate(indices)
        ],
        "predictions": [
            {
                "thread_id": test_data[i]['thread_id'],
                "attack_type": test_data[i]['attack_type'],
                "true_label": int(y_test[i]),
                "predicted_label": int(y_pred[i]),
                "attack_probability": round(float(y_proba[i]), 4),
                "correct": bool(y_pred[i] == y_test[i])
            }
            for i in range(len(y_test))
        ]
    }
    
    with open('baseline_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved: baseline_results.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("RANDOM FOREST BASELINE - THREAD-LEVEL PHISHING DETECTION")
    print("="*70)
    print("\nModel: Random Forest (Pedregosa et al., 2011)")
    print("Features: 12 handcrafted thread-level features")
    print("Task: Binary classification (legitimate=0 vs attack=1)")
    
    # Load data
    print("\n📂 Loading feature data...")
    X_train, y_train, train_data = load_features('train_features.json')
    X_test, y_test, test_data = load_features('test_features.json')
    
    print(f"  Train: {len(X_train)} threads (Legit: {sum(y_train==0)}, Attack: {sum(y_train==1)})")
    print(f"  Test:  {len(X_test)} threads (Legit: {sum(y_test==0)}, Attack: {sum(y_test==1)})")
    
    # Train
    print("\n🔧 Training Random Forest...")
    rf = train_model(X_train, y_train)
    print("  Done!")
    
    # Evaluate
    y_pred, y_proba = evaluate_model(rf, X_test, y_test, test_data)
    
    # Save
    save_results(rf, y_test, y_pred, y_proba, test_data)
    
    print("\n" + "="*70)
    print("✅ BASELINE MODEL COMPLETE")
    print("="*70)
    print("\n📌 Next: Train thread-aware BERT model for comparison")


if __name__ == '__main__':
    main()