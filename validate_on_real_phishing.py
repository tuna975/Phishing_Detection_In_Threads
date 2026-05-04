"""
Validation on Real Phishing Emails
====================================
Tests the trained DistilBERT model against:
  1. Real phishing/fraud emails (Label=1) from enron_data_fraud_labeled.csv
  2. Real legitimate emails (Label=0) sampled from the same CSV

This tells us whether the 100% training accuracy reflects genuine phishing
detection or is just picking up LLM-generated text vs. 2001 human email style.
"""

import json
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH       = "enron_data_fraud_labeled.csv"
MODEL_PATH     = "bert_phishing_model"
MAX_LENGTH     = 512
BATCH_SIZE     = 16
N_LEGIT_SAMPLE = 500   # how many legit emails to sample for reference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================================
# DATA LOADING
# ============================================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:3000]  # cap to avoid tokeniser overflow


def load_emails(csv_path):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Total rows: {len(df)}")
    print(f"  Label=1 (phishing): {(df['Label'] == 1).sum()}")
    print(f"  Label=0 (legit):    {(df['Label'] == 0).sum()}")

    phish = df[df['Label'] == 1].copy()
    legit = df[df['Label'] == 0].sample(n=N_LEGIT_SAMPLE, random_state=42).copy()

    return phish, legit


def emails_to_records(df, label):
    """Build simple single-email 'threads' from individual emails."""
    records = []
    for _, row in df.iterrows():
        subject = clean_text(str(row.get('Subject', '')))
        body    = clean_text(str(row.get('Body', '')))
        sender  = str(row.get('From', ''))
        text    = f"Subject: {subject} [SEP] {body}".strip()
        if len(text.split()) >= 5:
            records.append({'text': text, 'label': label, 'sender': sender, 'subject': subject})
    return records


# ============================================================================
# THREAD GROUPING (optional — groups emails with same subject)
# ============================================================================

def normalize_subject(subj):
    """Strip Re:/Fwd: prefixes for grouping."""
    return re.sub(r'^(re|fwd|fw)\s*:\s*', '', subj.lower().strip())


def group_into_threads(df, label):
    """Group individual emails into pseudo-threads by subject."""
    groups = defaultdict(list)
    for _, row in df.iterrows():
        subj_key = normalize_subject(str(row.get('Subject', '')))
        body     = clean_text(str(row.get('Body', '')))
        if body:
            groups[subj_key].append(body)

    records = []
    for subj_key, bodies in groups.items():
        text = " [SEP] ".join(bodies[:6])  # cap at 6 emails per thread
        if len(text.split()) >= 5:
            records.append({'text': text, 'label': label, 'subject': subj_key})
    return records


# ============================================================================
# DATASET
# ============================================================================

class EmailDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.encodings = tokenizer(
            [r['text'] for r in records],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )
        self.labels = torch.tensor([r['label'] for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx],
        }


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            labels    = batch['labels']

            out   = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = torch.softmax(out.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of class=1 (attack)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 65)
    print("VALIDATION: DistilBERT on Real Phishing Emails")
    print("=" * 65)

    # --- Load CSV ---
    phish_df, legit_df = load_emails(CSV_PATH)

    # --- Build records (single-email threads) ---
    print("\nBuilding single-email records...")
    phish_records = emails_to_records(phish_df, label=1)
    legit_records = emails_to_records(legit_df, label=0)
    print(f"  Real phishing emails usable : {len(phish_records)}")
    print(f"  Real legit emails sampled   : {len(legit_records)}")

    # --- Also try thread-grouped version for phishing ---
    print("\nGrouping phishing emails into pseudo-threads by subject...")
    phish_threads = group_into_threads(phish_df, label=1)
    print(f"  Pseudo-threads formed       : {len(phish_threads)}")

    # --- Load model ---
    print(f"\nLoading model from {MODEL_PATH} ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    # ---- EXPERIMENT 1: Individual phishing emails ----
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: Individual real phishing emails (Label=1 from CSV)")
    print("=" * 65)
    ds1   = EmailDataset(phish_records, tokenizer, MAX_LENGTH)
    dl1   = DataLoader(ds1, batch_size=BATCH_SIZE, shuffle=False)
    y1, p1, prob1 = run_inference(model, dl1)

    detected = (p1 == 1).sum()
    total    = len(y1)
    print(f"\n  Total real phishing emails tested : {total}")
    print(f"  Detected as attack (label=1)      : {detected} ({detected/total*100:.1f}%)")
    print(f"  Missed  (classified as legit)     : {total-detected} ({(total-detected)/total*100:.1f}%)")
    print(f"  Mean attack probability score     : {prob1.mean():.4f}")

    # ---- EXPERIMENT 2: Sampled legit emails ----
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Real legitimate emails (Label=0 from CSV, n=500)")
    print("=" * 65)
    ds2   = EmailDataset(legit_records, tokenizer, MAX_LENGTH)
    dl2   = DataLoader(ds2, batch_size=BATCH_SIZE, shuffle=False)
    y2, p2, prob2 = run_inference(model, dl2)

    fp = (p2 == 1).sum()
    total2 = len(y2)
    print(f"\n  Total real legit emails tested    : {total2}")
    print(f"  False positives (flagged as attack): {fp} ({fp/total2*100:.1f}%)")
    print(f"  Correct (legit classified legit)  : {total2-fp} ({(total2-fp)/total2*100:.1f}%)")
    print(f"  Mean attack probability score     : {prob2.mean():.4f}")

    # ---- EXPERIMENT 3: Thread-grouped phishing ----
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: Pseudo-threads (phishing emails grouped by subject)")
    print("=" * 65)
    ds3   = EmailDataset(phish_threads, tokenizer, MAX_LENGTH)
    dl3   = DataLoader(ds3, batch_size=BATCH_SIZE, shuffle=False)
    y3, p3, prob3 = run_inference(model, dl3)

    detected3 = (p3 == 1).sum()
    total3    = len(y3)
    print(f"\n  Total pseudo-threads tested       : {total3}")
    print(f"  Detected as attack (label=1)      : {detected3} ({detected3/total3*100:.1f}%)")
    print(f"  Missed                            : {total3-detected3} ({(total3-detected3)/total3*100:.1f}%)")
    print(f"  Mean attack probability score     : {prob3.mean():.4f}")

    # ---- COMBINED CLASSIFICATION REPORT ----
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

    # ---- VERDICT ----
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    train_acc = 1.0  # from our earlier run
    real_detect_rate = detected / total

    print(f"\n  Training accuracy (LLM-generated attacks): ~100%")
    print(f"  Detection on REAL phishing emails        : {real_detect_rate*100:.1f}%")
    print(f"  False-positive rate on real legit        : {fp/total2*100:.1f}%")

    if real_detect_rate < 0.60:
        verdict = (
            "LOW generalization. Model likely learned LLM text style rather than "
            "phishing semantics. The 100% training accuracy is a confound."
        )
    elif real_detect_rate < 0.80:
        verdict = (
            "MODERATE generalization. Model picks up some real phishing signals "
            "but is not robust — LLM-text style likely contributes to training accuracy."
        )
    else:
        verdict = (
            "GOOD generalization. Model detects real phishing emails well, "
            "suggesting it has learned meaningful phishing patterns beyond LLM style."
        )
    print(f"\n  {verdict}")

    # Save results
    results = {
        "experiment_1_individual": {
            "total": int(total), "detected": int(detected),
            "detection_rate": float(detected / total),
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
    import json
    with open("validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to: validation_results.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
