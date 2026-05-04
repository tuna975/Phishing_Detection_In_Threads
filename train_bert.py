"""
Thread-Level Phishing Detection — DistilBERT Fine-Tuning
=========================================================
Author : Anshuman Burman (2022BCS-009)
Supervisor : Dr. Debanjan Sadhya
Institution: ABV-IIITM Gwalior

Approach:
  All emails in a thread are concatenated into one text block,
  then fed to DistilBERT for binary classification.

  Separator token [SEP] is inserted between individual emails so the
  model sees email boundaries even after concatenation.

  Input format:
    [CLS] <email_1_body> [SEP] <email_2_body> [SEP] ... [SEP]

  Label: 0 = legitimate, 1 = phishing/attack

Model: distilbert-base-uncased
  - 40% smaller, 60% faster than bert-base
  - Retains 97% of BERT performance on classification tasks
  - Practical choice for training; runs on RTX 3050 GPU

References:
  - Devlin et al. (2019) - BERT
  - Sanh et al. (2019)   - DistilBERT
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

CFG = {
    "model_name":    "distilbert-base-uncased",
    "max_length":    512,          # BERT token limit
    "batch_size":    16,           # reduce to 8 if RAM is tight
    "epochs":        5,
    "lr":            2e-5,         # standard fine-tuning LR for BERT-family
    "warmup_ratio":  0.1,          # 10% of steps for LR warmup
    "weight_decay":  0.01,
    "seed":          42,
    "save_path":     "bert_phishing_model",
    "results_path":  "bert_results.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ============================================================================
# DATASET
# ============================================================================

def build_thread_text(thread):
    """
    Concatenate all email bodies with [SEP] between them.
    This lets DistilBERT see email boundaries.
    """
    bodies = [
        e.get("body", "").strip()
        for e in thread.get("emails", [])
        if e.get("body", "").strip()
    ]
    return " [SEP] ".join(bodies)


class ThreadDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length):
        self.samples = []
        threads = json.load(open(filepath, encoding="utf-8"))
        for thread in threads:
            text  = build_thread_text(thread)
            label = int(thread["label"])
            if text:
                self.samples.append((text, label))

        # Tokenize all at once (faster than per-item)
        texts  = [s[0] for s in self.samples]
        labels = [s[1] for s in self.samples]

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        # Gradient clipping — prevents exploding gradients during fine-tuning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    auc      = roc_auc_score(all_labels, all_probs)

    return avg_loss, acc, f1, auc, all_preds, all_probs, all_labels


# ============================================================================
# FULL TEST EVALUATION
# ============================================================================

def full_evaluation(model, test_loader, test_data, device):
    _, acc, f1, auc, preds, probs, labels = evaluate(model, test_loader, device)

    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)

    print("\n" + "="*65)
    print("TEST SET RESULTS")
    print("="*65)
    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=["Legitimate", "Attack"]))

    print("--- Confusion Matrix ---")
    print(f"                  Predicted")
    print(f"               Legit  Attack")
    print(f"  Actual Legit  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Actual Attack {cm[1][0]:5d}  {cm[1][1]:5d}")

    # Per attack type breakdown
    threads = json.load(open("test_dataset.json", encoding="utf-8"))
    print("\n--- Performance by Attack Type ---")
    attack_types = ["legitimate", "thread_hijacking", "relationship_exploitation",
                    "context_manipulation", "urgency_escalation"]
    for atype in attack_types:
        indices = [i for i, t in enumerate(threads) if t["attack_type"] == atype]
        if indices:
            correct = sum(1 for i in indices if preds[i] == labels[i])
            total   = len(indices)
            print(f"  {atype:<30}: {correct}/{total}  ({correct/total*100:.1f}%)")

    # Misclassified threads
    misclassified = [
        {
            "thread_id":   threads[i]["thread_id"],
            "attack_type": threads[i]["attack_type"],
            "true":        int(labels[i]),
            "predicted":   int(preds[i]),
            "prob":        round(float(probs[i]), 4),
        }
        for i in range(len(labels)) if preds[i] != labels[i]
    ]

    if misclassified:
        print(f"\n--- Misclassified Threads ({len(misclassified)}) ---")
        for m in misclassified[:20]:   # show first 20
            true_lbl = "Attack" if m["true"] == 1 else "Legit"
            pred_lbl = "Attack" if m["predicted"] == 1 else "Legit"
            print(f"  {m['thread_id']:<25} | {m['attack_type']:<28} | {true_lbl} → {pred_lbl}  (prob {m['prob']})")
        if len(misclassified) > 20:
            print(f"  ... and {len(misclassified)-20} more")
    else:
        print("\n  No misclassifications on test set.")

    return {
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "auc_roc":    round(auc,  4),
        "confusion_matrix": {
            "true_negative":  int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive":  int(cm[1][1]),
        },
        "misclassified_count": len(misclassified),
    }


# ============================================================================
# COMPARE WITH RANDOM FOREST BASELINE
# ============================================================================

def compare_with_baseline(bert_metrics):
    baseline_path = "baseline_results.json"
    if not os.path.exists(baseline_path):
        print("\n  (baseline_results.json not found — skipping comparison)")
        return

    baseline = json.load(open(baseline_path, encoding="utf-8"))
    rf = baseline["metrics"]

    print("\n" + "="*65)
    print("BERT vs RANDOM FOREST BASELINE")
    print("="*65)
    print(f"\n  {'Metric':<15} {'Random Forest':>15} {'DistilBERT':>15} {'Delta':>10}")
    print("  " + "-"*55)

    for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
        rf_val   = rf.get(metric, 0)
        bert_val = bert_metrics.get(metric, 0)
        delta    = bert_val - rf_val
        sign     = "+" if delta >= 0 else ""
        print(f"  {metric:<15} {rf_val:>15.4f} {bert_val:>15.4f} {sign+f'{delta:.4f}':>10}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(CFG["seed"])

    print("="*65)
    print("DISTILBERT FINE-TUNING — THREAD-LEVEL PHISHING DETECTION")
    print("="*65)
    print(f"\n  Model  : {CFG['model_name']}")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {CFG['epochs']}")
    print(f"  LR     : {CFG['lr']}")
    print(f"  Batch  : {CFG['batch_size']}")
    print(f"  MaxLen : {CFG['max_length']} tokens")

    # ---- Tokenizer & Model ----
    print("\n[1/5] Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(CFG["model_name"])
    model     = DistilBertForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels=2,
    ).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ---- Datasets ----
    print("\n[2/5] Loading and tokenizing datasets...")
    train_ds = ThreadDataset("train_dataset.json", tokenizer, CFG["max_length"])
    val_ds   = ThreadDataset("val_dataset.json",   tokenizer, CFG["max_length"])
    test_ds  = ThreadDataset("test_dataset.json",  tokenizer, CFG["max_length"])

    print(f"  Train : {len(train_ds)} threads")
    print(f"  Val   : {len(val_ds)} threads")
    print(f"  Test  : {len(test_ds)} threads")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False)

    # ---- Class weights (handle 1000 legit vs 400 attack imbalance) ----
    labels      = [s[1] for s in train_ds.samples]
    n_legit     = labels.count(0)
    n_attack    = labels.count(1)
    weight_legit  = len(labels) / (2 * n_legit)
    weight_attack = len(labels) / (2 * n_attack)
    class_weights = torch.tensor([weight_legit, weight_attack], dtype=torch.float).to(DEVICE)
    print(f"\n  Class weights — Legit: {weight_legit:.3f}  Attack: {weight_attack:.3f}")

    # Override model loss with weighted cross entropy
    model.config.problem_type = "single_label_classification"

    # ---- Optimizer & Scheduler ----
    print("\n[3/5] Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    total_steps  = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"  Total steps  : {total_steps}")
    print(f"  Warmup steps : {warmup_steps}")

    # ---- Training Loop ----
    print("\n[4/5] Training...")
    print(f"\n  {'Epoch':<7} {'Train Loss':<12} {'Train Acc':<12} {'Train F1':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<10} {'Val AUC'}")
    print("  " + "-"*90)

    best_val_f1   = -1
    best_epoch    = -1
    history       = []

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        vl_loss, vl_acc, vl_f1, vl_auc, _, _, _ = evaluate(model, val_loader, DEVICE)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4), "train_f1": round(tr_f1, 4),
            "val_loss":   round(vl_loss, 4), "val_acc":   round(vl_acc, 4), "val_f1":   round(vl_f1, 4),
            "val_auc":    round(vl_auc, 4),
        })

        marker = " <-- best" if vl_f1 > best_val_f1 else ""
        print(f"  {epoch:<7} {tr_loss:<12.4f} {tr_acc:<12.4f} {tr_f1:<12.4f} "
              f"{vl_loss:<12.4f} {vl_acc:<12.4f} {vl_f1:<10.4f} {vl_auc:.4f}{marker}")

        # Save best model by validation F1
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch  = epoch
            model.save_pretrained(CFG["save_path"])
            tokenizer.save_pretrained(CFG["save_path"])

    print(f"\n  Best model: epoch {best_epoch}  (val F1 = {best_val_f1:.4f})")

    # ---- Test Evaluation ----
    print("\n[5/5] Loading best model and evaluating on test set...")
    best_model = DistilBertForSequenceClassification.from_pretrained(CFG["save_path"]).to(DEVICE)
    test_metrics = full_evaluation(best_model, test_loader, test_ds, DEVICE)

    # ---- Compare with RF baseline ----
    compare_with_baseline(test_metrics)

    # ---- Save results ----
    results = {
        "model":   CFG["model_name"],
        "config":  CFG,
        "device":  str(DEVICE),
        "best_epoch": best_epoch,
        "training_history": history,
        "test_metrics": test_metrics,
    }
    with open(CFG["results_path"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Model saved : {CFG['save_path']}/")
    print(f"  Results saved: {CFG['results_path']}")
    print("\n" + "="*65)
    print("TRAINING COMPLETE")
    print("="*65)


if __name__ == "__main__":
    main()
