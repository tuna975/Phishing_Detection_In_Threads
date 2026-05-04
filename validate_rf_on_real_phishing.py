"""
Validate Random Forest Baseline on Real Phishing Emails
=========================================================
Mirrors validate_on_real_phishing.py but uses the saved Random Forest model
(random_forest_model.pkl) and the 12 handcrafted thread-level features.

Experiments:
  1. Individual real phishing emails as single-email threads
  2. 500 sampled real legit emails as single-email threads
  3. Phishing emails grouped into pseudo-threads by subject
"""

import json
import pickle
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH        = "enron_data_fraud_labeled.csv"
MODEL_PATH      = "random_forest_model.pkl"
N_LEGIT_SAMPLE  = 500

FEATURE_COLS = [
    'response_time_variance', 'thread_length_days', 'timestamp_anomalies',
    'tone_shift', 'formality_variance', 'urgency_escalation',
    'reference_accuracy', 'entity_consistency', 'topic_coherence',
    'fabricated_history_score', 'relationship_velocity', 'cross_reference_count'
]

# ============================================================================
# FEATURE EXTRACTION (copied exactly from extract_features.py)
# ============================================================================

def parse_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        if 'T' in date_str:
            date_str = date_str.split('.')[0]
            return datetime.fromisoformat(date_str)
        return None
    except Exception:
        return None


def extract_temporal_features(thread):
    emails = thread['emails']
    dates = [parse_date(e.get('date', '')) for e in emails]
    dates = [d for d in dates if d]
    if len(dates) < 2:
        return {'response_time_variance': 0, 'thread_length_days': 0, 'timestamp_anomalies': 0}
    dates.sort()
    response_times = [(dates[i+1] - dates[i]).total_seconds() / 3600
                      for i in range(len(dates) - 1)]
    return {
        'response_time_variance': np.var(response_times) if len(response_times) > 1 else 0,
        'thread_length_days':     (dates[-1] - dates[0]).total_seconds() / 86400,
        'timestamp_anomalies':    sum(1 for d in dates if d.hour >= 23 or d.hour <= 5) / len(dates),
    }


def extract_linguistic_features(thread):
    emails = thread['emails']
    urgency_words = [
        'urgent', 'asap', 'immediately', 'emergency', 'critical', 'now',
        'rush', 'hurry', 'quick', 'fast', 'deadline', 'crisis', 'catastrophic',
        'disaster', 'panic', 'desperate'
    ]
    formal_words   = ['please', 'thank', 'regards', 'sincerely', 'respectfully']
    informal_words = ['hey', 'gonna', 'wanna', 'yeah', 'ok', 'btw']

    urgency_scores   = []
    formality_scores = []

    for email in emails:
        body = email.get('body', '').lower()
        urgency_scores.append(sum(1 for w in urgency_words if w in body))
        formality_scores.append(
            sum(1 for w in formal_words if w in body) -
            sum(1 for w in informal_words if w in body)
        )

    return {
        'tone_shift':          abs(formality_scores[-1] - formality_scores[0]) if len(formality_scores) > 1 else 0,
        'formality_variance':  np.var(formality_scores) if len(formality_scores) > 1 else 0,
        'urgency_escalation':  urgency_scores[-1] - urgency_scores[0] if len(urgency_scores) > 1 else 0,
    }


def extract_coherence_features(thread):
    emails = thread['emails']
    reference_words = ['as we discussed', 'as mentioned', 'per our', 'following up',
                       'our call', 'our meeting', 'we talked', 'you said']

    ref_count = sum(
        sum(1 for phrase in reference_words if phrase in e.get('body', '').lower())
        for e in emails
    )

    addrs = set()
    for e in emails:
        if e.get('from'): addrs.add(e['from'].lower())
        if e.get('to'):   addrs.add(e['to'].lower())

    subjects     = [e.get('subject', '').lower() for e in emails]
    base_subjects = [re.sub(r'^(re:|fwd:)\s*', '', s).strip() for s in subjects]
    unique_subj  = len(set(base_subjects))

    return {
        'reference_accuracy':  ref_count / len(emails) if emails else 0,
        'entity_consistency':  1.0 / len(addrs) if addrs else 1.0,
        'topic_coherence':     1.0 / unique_subj if unique_subj > 0 else 1.0,
    }


def extract_context_features(thread):
    emails = thread['emails']
    fabrication_phrases = [
        'as we discussed', 'per our conversation', 'following our meeting',
        'as mentioned in our call', 'from our discussion', 'per our agreement',
        'as we agreed', 'from yesterday', 'last week', 'our session'
    ]
    cross_ref_words = ['forward', 'fwd', 'attachment', 'attached', 'cc', 'bcc', 'replied']

    fab_count = sum(
        sum(1 for phrase in fabrication_phrases if phrase in e.get('body', '').lower())
        for e in emails
    )

    formality_changes = sum(
        1 for i in range(1, len(emails))
        if ('dear' in emails[i-1].get('body', '').lower() and 'hey' in emails[i].get('body', '').lower())
        or ('regards' in emails[i-1].get('body', '').lower() and 'regards' not in emails[i].get('body', '').lower())
    )

    cross_ref = sum(
        sum(1 for w in cross_ref_words if w in e.get('body', '').lower())
        for e in emails
    )

    return {
        'fabricated_history_score': fab_count / len(emails) if emails else 0,
        'relationship_velocity':    formality_changes / (len(emails) - 1) if len(emails) > 1 else 0,
        'cross_reference_count':    cross_ref / len(emails) if emails else 0,
    }


def extract_features(thread):
    feats = {}
    feats.update(extract_temporal_features(thread))
    feats.update(extract_linguistic_features(thread))
    feats.update(extract_coherence_features(thread))
    feats.update(extract_context_features(thread))
    return [feats[c] for c in FEATURE_COLS]


# ============================================================================
# CSV → THREAD CONVERSION
# ============================================================================

def row_to_email(row):
    """Convert a CSV row to an email dict the feature extractor understands."""
    return {
        'date':    str(row.get('Date', '') or ''),
        'from':    str(row.get('From', '') or ''),
        'to':      str(row.get('To', '')   or ''),
        'subject': str(row.get('Subject', '') or ''),
        'body':    str(row.get('Body', '')    or ''),
    }


def df_to_single_email_threads(df, label):
    """Each row becomes a one-email 'thread'."""
    threads = []
    for i, (_, row) in enumerate(df.iterrows()):
        email = row_to_email(row)
        if len(email['body'].split()) < 5:
            continue
        threads.append({
            'thread_id': f"real_{label}_{i}",
            'label':     label,
            'emails':    [email],
        })
    return threads


def normalize_subject(subj):
    return re.sub(r'^(re|fwd|fw)\s*:\s*', '', subj.lower().strip())


def df_to_grouped_threads(df, label):
    """Group rows into pseudo-threads by subject."""
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = normalize_subject(str(row.get('Subject', '') or ''))
        groups[key].append(row_to_email(row))

    threads = []
    for i, (subj_key, emails) in enumerate(groups.items()):
        # Cap at 6 emails, skip tiny groups
        emails = emails[:6]
        combined_body = " ".join(e['body'] for e in emails)
        if len(combined_body.split()) < 5:
            continue
        threads.append({
            'thread_id': f"grp_{label}_{i}",
            'label':     label,
            'emails':    emails,
        })
    return threads


# ============================================================================
# INFERENCE
# ============================================================================

def predict(rf, threads):
    X      = np.array([extract_features(t) for t in threads])
    labels = np.array([t['label'] for t in threads])
    preds  = rf.predict(X)
    probs  = rf.predict_proba(X)[:, 1]
    return labels, preds, probs


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("VALIDATION: Random Forest on Real Phishing Emails")
    print("=" * 65)

    # Load CSV
    print(f"\nLoading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    phish_df = df[df['Label'] == 1].copy()
    legit_df  = df[df['Label'] == 0].sample(n=N_LEGIT_SAMPLE, random_state=42).copy()
    print(f"  Phishing emails : {len(phish_df)}")
    print(f"  Legit sample    : {len(legit_df)}")

    # Load model
    print(f"\nLoading {MODEL_PATH} ...")
    with open(MODEL_PATH, 'rb') as f:
        rf = pickle.load(f)
    print("  Loaded.")

    # ---- EXPERIMENT 1: Individual phishing emails ----
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: Individual real phishing emails (Label=1)")
    print("=" * 65)
    phish_single = df_to_single_email_threads(phish_df, label=1)
    y1, p1, prob1 = predict(rf, phish_single)
    detected1 = (p1 == 1).sum()
    total1    = len(y1)
    print(f"\n  Total real phishing emails tested : {total1}")
    print(f"  Detected as attack (label=1)      : {detected1} ({detected1/total1*100:.1f}%)")
    print(f"  Missed                            : {total1-detected1} ({(total1-detected1)/total1*100:.1f}%)")
    print(f"  Mean attack probability score     : {prob1.mean():.4f}")

    # ---- EXPERIMENT 2: Legit emails ----
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Real legitimate emails (Label=0, n=500)")
    print("=" * 65)
    legit_single = df_to_single_email_threads(legit_df, label=0)
    y2, p2, prob2 = predict(rf, legit_single)
    fp     = (p2 == 1).sum()
    total2 = len(y2)
    print(f"\n  Total real legit emails tested    : {total2}")
    print(f"  False positives (flagged attack)  : {fp} ({fp/total2*100:.1f}%)")
    print(f"  Correct (legit classified legit)  : {total2-fp} ({(total2-fp)/total2*100:.1f}%)")
    print(f"  Mean attack probability score     : {prob2.mean():.4f}")

    # ---- EXPERIMENT 3: Grouped pseudo-threads ----
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: Phishing pseudo-threads grouped by subject")
    print("=" * 65)
    phish_grouped = df_to_grouped_threads(phish_df, label=1)
    y3, p3, prob3 = predict(rf, phish_grouped)
    detected3 = (p3 == 1).sum()
    total3    = len(y3)
    print(f"\n  Total pseudo-threads tested       : {total3}")
    print(f"  Detected as attack (label=1)      : {detected3} ({detected3/total3*100:.1f}%)")
    print(f"  Missed                            : {total3-detected3} ({(total3-detected3)/total3*100:.1f}%)")
    print(f"  Mean attack probability score     : {prob3.mean():.4f}")

    # ---- COMBINED REPORT ----
    print("\n" + "=" * 65)
    print("COMBINED REPORT (Experiment 1 phishing + Experiment 2 legit)")
    print("=" * 65)
    y_all = np.concatenate([y1, y2])
    p_all = np.concatenate([p1, p2])
    print("\n" + classification_report(y_all, p_all, target_names=["Legitimate", "Attack"]))

    cm = confusion_matrix(y_all, p_all)
    print("Confusion Matrix:")
    print(f"  {'':20} Pred Legit   Pred Attack")
    print(f"  {'True Legit':20} {cm[0,0]:>10}   {cm[0,1]:>10}")
    print(f"  {'True Attack':20} {cm[1,0]:>10}   {cm[1,1]:>10}")

    # ---- HEAD-TO-HEAD COMPARISON ----
    print("\n" + "=" * 65)
    print("HEAD-TO-HEAD: Random Forest vs DistilBERT (from previous run)")
    print("=" * 65)
    print(f"\n  {'Metric':<40} {'RF':>8} {'DistilBERT':>12}")
    print(f"  {'-'*62}")
    print(f"  {'Real phishing detection rate':40} {detected1/total1*100:>7.1f}% {'41.3%':>12}")
    print(f"  {'False-positive rate (legit)':40} {fp/total2*100:>7.1f}% {'7.4%':>12}")
    print(f"  {'Mean attack prob (phishing)':40} {prob1.mean():>8.4f} {'0.4100':>12}")
    print(f"  {'Mean attack prob (legit)':40} {prob2.mean():>8.4f} {'0.0769':>12}")

    # ---- VERDICT ----
    real_detect_rate = detected1 / total1
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    print(f"\n  Training accuracy (LLM-generated attacks): ~100%  (both models)")
    print(f"  RF detection on REAL phishing emails     : {real_detect_rate*100:.1f}%")
    print(f"  DistilBERT detection on REAL phishing    : 41.3%")

    if real_detect_rate > 0.60:
        verdict = (
            "RF generalizes BETTER than DistilBERT to real phishing. "
            "The handcrafted features capture meaningful semantic patterns, "
            "while DistilBERT's accuracy was largely driven by LLM-text style."
        )
    elif abs(real_detect_rate - 0.413) < 0.05:
        verdict = (
            "RF and DistilBERT perform similarly on real phishing — "
            "both struggle with out-of-distribution data, suggesting the dataset "
            "confound affects both models."
        )
    else:
        verdict = (
            "Both models generalize poorly to real phishing, confirming "
            "the LLM-text confound. RF features are more interpretable but "
            "do not resolve the distribution mismatch."
        )
    print(f"\n  {verdict}")

    # Save
    results = {
        "experiment_1_individual": {
            "total": int(total1), "detected": int(detected1),
            "detection_rate": float(detected1 / total1),
            "mean_attack_prob": float(prob1.mean()),
        },
        "experiment_2_legit": {
            "total": int(total2), "false_positives": int(fp),
            "fp_rate": float(fp / total2),
            "mean_attack_prob": float(prob2.mean()),
        },
        "experiment_3_threads": {
            "total": int(total3), "detected": int(detected3),
            "detection_rate": float(detected3 / total3),
            "mean_attack_prob": float(prob3.mean()),
        },
    }
    with open("validation_rf_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to: validation_rf_results.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
