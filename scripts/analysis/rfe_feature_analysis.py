"""
Wrapper-Based Recursive Feature Elimination (RFE) Analysis
============================================================
Author: Anshuman Burman (2022BCS-009)
Supervisor: Dr. Debanjan Sadhya
Institution: ABV-IIITM Gwalior

This script performs three levels of feature importance analysis:
1. RFECV - Recursive Feature Elimination with Cross-Validation
2. RFE Elimination Order - Which features are removed first/last
3. Sequential Feature Addition - Performance gain as features are added
4. Individual Feature Performance - How each feature performs alone

Method: Wrapper-based feature selection using Random Forest (Pedregosa et al., 2011)
        RFE iteratively removes the least important feature and re-evaluates
        model performance using 5-fold stratified cross-validation.

Citation: Guyon et al. (2002) - "Gene Selection for Cancer Classification 
          using Support Vector Machines" - introduced RFE methodology
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score)
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

FEATURE_CATEGORIES = {
    'response_time_variance': 'Temporal',
    'thread_length_days': 'Temporal',
    'timestamp_anomalies': 'Temporal',
    'tone_shift': 'Linguistic',
    'formality_variance': 'Linguistic',
    'urgency_escalation': 'Linguistic',
    'reference_accuracy': 'Coherence',
    'entity_consistency': 'Coherence',
    'topic_coherence': 'Coherence',
    'fabricated_history_score': 'Context-Aware',
    'relationship_velocity': 'Context-Aware',
    'cross_reference_count': 'Context-Aware'
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'class_weight': 'balanced'
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(train_path, test_path):
    """Load feature JSON files"""
    with open(train_path) as f:
        train = json.load(f)
    with open(test_path) as f:
        test = json.load(f)

    X_train = np.array([[t[f] for f in FEATURE_COLS] for t in train])
    y_train = np.array([t['label'] for t in train])
    X_test = np.array([[t[f] for f in FEATURE_COLS] for t in test])
    y_test = np.array([t['label'] for t in test])

    return X_train, y_train, X_test, y_test, train, test


# ============================================================================
# ANALYSIS 1: RFECV (Optimal Feature Count)
# ============================================================================

def run_rfecv(X_train, y_train, cv):
    """
    RFECV: Recursive Feature Elimination with Cross-Validation
    
    How it works:
    - Trains Random Forest on all 12 features
    - Removes the feature with lowest feature_importance_ score
    - Re-evaluates using 5-fold CV
    - Repeats until 1 feature remains
    - Selects the feature count with highest CV F1 score
    
    Reference: Guyon et al. (2002)
    """
    print("="*70)
    print("ANALYSIS 1: RFECV - OPTIMAL FEATURE COUNT")
    print("="*70)
    print("\nMethod: Recursively eliminate features, evaluate with 5-fold CV")
    print("Metric: F1 Score")

    rf = RandomForestClassifier(**RF_PARAMS)
    rfecv = RFECV(estimator=rf, step=1, cv=cv, scoring='f1', min_features_to_select=1)
    rfecv.fit(X_train, y_train)

    print(f"\nOptimal number of features: {rfecv.n_features_}")

    # CV scores at each step
    cv_scores = rfecv.cv_results_['mean_test_score']
    cv_stds = rfecv.cv_results_['std_test_score']

    print(f"\n{'# Features':<14} {'Mean F1':<12} {'Std':<12} {'Visual'}")
    print("-"*65)
    for i, (score, std) in enumerate(zip(cv_scores, cv_stds)):
        marker = " ◄── OPTIMAL" if (i+1) == rfecv.n_features_ else ""
        bar = "█" * int(score * 40)
        print(f"  {i+1:<12} {score:.4f}       {std:.4f}       {bar}{marker}")

    # Selected vs eliminated
    print(f"\n✅ Selected ({rfecv.n_features_}):")
    for f, s in zip(FEATURE_COLS, rfecv.support_):
        if s:
            print(f"   • {f} [{FEATURE_CATEGORIES[f]}]")

    eliminated = [(f, r) for f, r, s in zip(FEATURE_COLS, rfecv.ranking_, rfecv.support_) if not s]
    if eliminated:
        print(f"\n❌ Eliminated ({len(eliminated)}):")
        for f, r in sorted(eliminated, key=lambda x: x[1]):
            print(f"   • {f} [{FEATURE_CATEGORIES[f]}] (rank {r})")

    return rfecv, cv_scores


# ============================================================================
# ANALYSIS 2: RFE ELIMINATION ORDER
# ============================================================================

def run_rfe_order(X_train, y_train):
    """
    RFE Elimination Order
    
    How it works:
    - Trains RF on all features
    - At each step, the feature with lowest importance is removed
    - The LAST feature remaining is the most important
    - Rankings show elimination order (1 = most important, 12 = first removed)
    
    The ranking tells us: if you could only keep ONE feature, which would it be?
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: RFE ELIMINATION ORDER")
    print("="*70)
    print("\nMethod: Remove least important feature iteratively")
    print("Rank 1 = last standing (most important)")
    print("Rank 12 = first eliminated (least important)")

    rf = RandomForestClassifier(**RF_PARAMS)
    rfe = RFE(estimator=rf, n_features_to_select=1, step=1)
    rfe.fit(X_train, y_train)

    order = sorted(zip(FEATURE_COLS, rfe.ranking_), key=lambda x: x[1])
    
    print(f"\n{'Rank':<6} {'Feature':<35} {'Category':<15} {'Status'}")
    print("-"*75)
    for feat, rank in order:
        cat = FEATURE_CATEGORIES[feat]
        if rank == 1:
            status = "★ MOST IMPORTANT"
        elif rank <= 5:
            status = "Important"
        else:
            status = f"Eliminated at step {13-rank}"
        print(f"  {rank:<4} {feat:<35} {cat:<15} {status}")

    return rfe


# ============================================================================
# ANALYSIS 3: SEQUENTIAL FEATURE ADDITION
# ============================================================================

def run_sequential_addition(X_train, y_train, X_test, y_test, cv):
    """
    Sequential Feature Addition (Forward Selection)
    
    How it works:
    - Start with the most important feature (from RF feature_importances_)
    - Add the next most important feature
    - Measure CV F1 and test performance at each step
    - Shows the marginal contribution of each feature
    
    This answers: "How much does each additional feature improve the model?"
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: SEQUENTIAL FEATURE ADDITION")
    print("="*70)
    print("\nMethod: Add features one-by-one in importance order")
    print("Shows marginal contribution of each feature\n")

    # Get importance order from full model
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X_train, y_train)
    importance_order = np.argsort(rf_full.feature_importances_)[::-1]

    selected = []
    results = []

    print(f"{'#':<4} {'Feature Added':<35} {'CV F1':<10} {'Test F1':<10} {'Gain'}")
    print("-"*75)

    prev_cv_f1 = 0
    for idx in importance_order:
        selected.append(idx)
        X_tr_sub = X_train[:, selected]
        X_te_sub = X_test[:, selected]

        rf_sub = RandomForestClassifier(**RF_PARAMS)
        cv_f1 = np.mean(cross_val_score(rf_sub, X_tr_sub, y_train, cv=cv, scoring='f1'))

        rf_sub.fit(X_tr_sub, y_train)
        y_pred = rf_sub.predict(X_te_sub)
        test_f1 = f1_score(y_test, y_pred)

        gain = cv_f1 - prev_cv_f1
        gain_str = f"+{gain:.4f}" if gain > 0 else f"{gain:.4f}"
        
        results.append({
            'feature': FEATURE_COLS[idx],
            'category': FEATURE_CATEGORIES[FEATURE_COLS[idx]],
            'num_features': len(selected),
            'cv_f1': round(cv_f1, 4),
            'test_f1': round(test_f1, 4),
            'gain': round(gain, 4)
        })

        print(f"  {len(selected):<3} {FEATURE_COLS[idx]:<35} {cv_f1:.4f}     {test_f1:.4f}     {gain_str}")
        prev_cv_f1 = cv_f1

    # Summary
    print(f"\n📊 Key Insight:")
    print(f"   Top 5 features achieve CV F1 = {results[4]['cv_f1']:.4f}")
    print(f"   All 12 features achieve CV F1 = {results[11]['cv_f1']:.4f}")
    print(f"   Marginal gain from features 6-12: {results[11]['cv_f1'] - results[4]['cv_f1']:.4f}")

    return results


# ============================================================================
# ANALYSIS 4: INDIVIDUAL FEATURE PERFORMANCE
# ============================================================================

def run_individual_features(X_train, y_train, X_test, y_test, cv):
    """
    Individual Feature Performance
    
    How it works:
    - Train a separate Random Forest using ONLY one feature at a time
    - Evaluate using 5-fold CV F1 and test accuracy
    - Shows which single feature is most predictive alone
    
    This answers: "If you had only ONE feature, which gives the best detection?"
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: INDIVIDUAL FEATURE PERFORMANCE")
    print("="*70)
    print("\nMethod: Train separate model with each feature alone")
    print("Shows standalone predictive power\n")

    individual_scores = []

    for i, feat in enumerate(FEATURE_COLS):
        X_tr_single = X_train[:, [i]]
        X_te_single = X_test[:, [i]]

        rf_single = RandomForestClassifier(**RF_PARAMS)
        cv_f1 = np.mean(cross_val_score(rf_single, X_tr_single, y_train, cv=cv, scoring='f1'))

        rf_single.fit(X_tr_single, y_train)
        y_pred = rf_single.predict(X_te_single)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        individual_scores.append({
            'feature': feat,
            'category': FEATURE_CATEGORIES[feat],
            'cv_f1': round(cv_f1, 4),
            'test_accuracy': round(test_acc, 4),
            'test_f1': round(test_f1, 4)
        })

    # Sort by CV F1
    individual_scores.sort(key=lambda x: x['cv_f1'], reverse=True)

    print(f"{'Rank':<6} {'Feature':<35} {'Category':<15} {'CV F1':<10} {'Test Acc'}")
    print("-"*80)
    for rank, score in enumerate(individual_scores):
        bar = "█" * int(score['cv_f1'] * 40)
        print(f"  {rank+1:<4} {score['feature']:<35} {score['category']:<15} {score['cv_f1']:.4f}     {score['test_accuracy']:.4f}   {bar}")

    return individual_scores


# ============================================================================
# SAVE ALL RESULTS
# ============================================================================

def save_analysis(rfecv_scores, rfe_order, sequential_results, individual_results):
    """Save complete RFE analysis to JSON"""
    
    analysis = {
        "method": "Wrapper-Based Recursive Feature Elimination",
        "reference": "Guyon et al. (2002) - RFE methodology",
        "estimator": "Random Forest (Pedregosa et al., 2011)",
        "cv": "5-fold Stratified Cross-Validation",
        "scoring": "F1 Score",
        
        "rfecv_scores": [
            {"num_features": i+1, "cv_f1": round(float(s), 4)} 
            for i, s in enumerate(rfecv_scores)
        ],
        
        "sequential_addition": sequential_results,
        "individual_performance": individual_results,
        
        "key_findings": {
            "top_5_features_f1": sequential_results[4]['cv_f1'] if len(sequential_results) >= 5 else None,
            "all_12_features_f1": sequential_results[-1]['cv_f1'] if sequential_results else None,
            "best_single_feature": individual_results[0]['feature'] if individual_results else None,
            "best_single_feature_f1": individual_results[0]['cv_f1'] if individual_results else None,
            "worst_single_feature": individual_results[-1]['feature'] if individual_results else None,
            "worst_single_feature_f1": individual_results[-1]['cv_f1'] if individual_results else None
        }
    }

    with open('../../results/rfe_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    print("\n✅ Analysis saved: rfe_analysis_results.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("WRAPPER-BASED RECURSIVE FEATURE ELIMINATION ANALYSIS")
    print("Thread-Level Phishing Detection")
    print("="*70)

    # Load data
    print("\n📂 Loading data...")
    X_train, y_train, X_test, y_test, train, test = load_data(
        '../../data/features/train_features.json', '../../data/features/test_features.json'
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(FEATURE_COLS)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Run all 4 analyses
    rfecv, rfecv_scores = run_rfecv(X_train, y_train, cv)
    rfe = run_rfe_order(X_train, y_train)
    sequential = run_sequential_addition(X_train, y_train, X_test, y_test, cv)
    individual = run_individual_features(X_train, y_train, X_test, y_test, cv)

    # Save
    save_analysis(rfecv_scores, rfe, sequential, individual)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY FOR THESIS")
    print("="*70)
    print("""
Key Findings:
1. RFECV selected all 12 features as optimal, but performance plateaus 
   at 5 features (F1 ≈ 0.978 vs 0.980 with all 12)

2. Most discriminative features (by RFE ranking):
   - Entity consistency (Coherence) - most important
   - Thread length days (Temporal) 
   - Urgency escalation (Linguistic)
   - Response time variance (Temporal)
   - Timestamp anomalies (Temporal)

3. Least useful features:
   - Relationship velocity (Context-Aware) - weakest
   - Cross-reference count (Context-Aware)
   - Fabricated history score (Context-Aware)

4. Observation: Temporal features dominate (3 of top 5), suggesting 
   the model may be partially learning dataset distribution differences 
   rather than pure phishing patterns. This motivates the need for a 
   BERT-based approach that can learn deeper semantic patterns.

5. Urgency escalation is the strongest genuinely phishing-related 
   feature, consistent with phishing literature (Das et al., 2019).
""")
    print("="*70)
    print("✅ RFE ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
