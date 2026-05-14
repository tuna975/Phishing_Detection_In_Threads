"""
Combine all attack thread JSON batches with legitimate Enron threads
Final dataset: 1400 threads (1000 legitimate + 400 attacks)
"""

import json
import os
from collections import defaultdict
import random

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def combine_dataset():
    """Combine all threads into final dataset"""
    
    print("="*60)
    print("COMBINING DATASET")
    print("="*60)
    
    all_threads = []
    
    # Load Thread Hijacking
    print("\n📂 Thread Hijacking:")
    for i in range(1, 7):
        filepath = f"../../data/raw/Dataset Attack/hijack_batch{i}.json"
        if os.path.exists(filepath):
            threads = load_json(filepath)
            all_threads.extend(threads)
            print(f"  ✓ {filepath}: {len(threads)} threads")
    
    # Load Relationship Exploitation
    print("\n📂 Relationship Exploitation:")
    for i in range(1, 5):
        filepath = f"../../data/raw/Dataset Attack/relationship_batch{i}.json"
        if os.path.exists(filepath):
            threads = load_json(filepath)
            all_threads.extend(threads)
            print(f"  ✓ {filepath}: {len(threads)} threads")
    
    # Load Context Manipulation
    print("\n📂 Context Manipulation:")
    for i in range(1, 5):
        filepath = f"../../data/raw/Dataset Attack/context_batch{i}.json"
        if os.path.exists(filepath):
            threads = load_json(filepath)
            all_threads.extend(threads)
            print(f"  ✓ {filepath}: {len(threads)} threads")
    
    # Load Urgency Escalation
    print("\n📂 Urgency Escalation:")
    for i in range(1, 5):
        filepath = f"../../data/raw/Dataset Attack/urgency_batch{i}.json"
        if os.path.exists(filepath):
            threads = load_json(filepath)
            all_threads.extend(threads)
            print(f"  ✓ {filepath}: {len(threads)} threads")
    
    # Load Legitimate threads
    print("\n📂 Legitimate Threads:")
    leg_path = "../../data/raw/cleaned_threads_v2/top_1000_for_dataset.json"
    if os.path.exists(leg_path):
        leg_data = load_json(leg_path)
        leg_threads = []
        for _, thread in leg_data.items():
            thread['label'] = 0
            thread['attack_type'] = 'legitimate'
            leg_threads.append(thread)
        all_threads.extend(leg_threads)
        print(f"  ✓ {leg_path}: {len(leg_threads)} threads")
    else:
        print(f"  ✗ ERROR: {leg_path} not found!")
        return
    
    # Statistics
    print("\n"+"="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    stats = defaultdict(int)
    email_counts = []
    
    for thread in all_threads:
        stats[thread['attack_type']] += 1
        stats[f"label_{thread['label']}"] += 1
        email_counts.append(thread['num_emails'])
    
    print(f"Total: {len(all_threads)} threads\n")
    print("By Attack Type:")
    for attack_type in ['thread_hijacking', 'relationship_exploitation', 'context_manipulation', 'urgency_escalation', 'legitimate']:
        count = stats[attack_type]
        if count > 0:
            pct = count/len(all_threads)*100
            print(f"  {attack_type}: {count} ({pct:.1f}%)")
    
    print(f"\nBy Label:")
    print(f"  Legitimate (0): {stats['label_0']} ({stats['label_0']/len(all_threads)*100:.1f}%)")
    print(f"  Attack (1): {stats['label_1']} ({stats['label_1']/len(all_threads)*100:.1f}%)")
    
    print(f"\nEmails per thread: {sum(email_counts)/len(email_counts):.1f} avg (range: {min(email_counts)}-{max(email_counts)})")
    
    # Save combined
    with open('../../data/splits/combined_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(all_threads, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved: combined_dataset.json ({len(all_threads)} threads)")
    
    # Create splits (stratified by attack_type)
    print("\n"+"="*60)
    print("CREATING SPLITS (70% train / 15% val / 15% test)")
    print("="*60)
    
    random.seed(42)
    by_type = defaultdict(list)
    for t in all_threads:
        by_type[t['attack_type']].append(t)
    
    train, val, test = [], [], []
    
    for attack_type, threads in by_type.items():
        random.shuffle(threads)
        n = len(threads)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        train.extend(threads[:n_train])
        val.extend(threads[n_train:n_train+n_val])
        test.extend(threads[n_train+n_val:])
        
        print(f"{attack_type}:")
        print(f"  Train: {len(threads[:n_train])} | Val: {len(threads[n_train:n_train+n_val])} | Test: {len(threads[n_train+n_val:])}")
    
    print(f"\nTOTAL: Train={len(train)} | Val={len(val)} | Test={len(test)}")
    
    # Save splits
    with open('../../data/splits/train_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open('../../data/splits/val_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    with open('../../data/splits/test_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Saved: train_dataset.json")
    print("✓ Saved: val_dataset.json")
    print("✓ Saved: test_dataset.json")
    
    print("\n"+"="*60)
    print("✅ DATASET READY FOR FEATURE EXTRACTION")
    print("="*60)

if __name__ == '__main__':
    combine_dataset()