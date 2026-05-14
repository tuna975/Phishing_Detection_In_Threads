"""
Step 4 V2: Comprehensive Deduplication with Timestamp Awareness
Removes duplicates and ensures proper chronological ordering
"""

import json
import os
from datetime import datetime
from collections import defaultdict
import hashlib

# Configuration
INPUT_FILE = "../../data/raw/extracted_threads_v2/high_quality_threads.json"
OUTPUT_DIR = "../../data/raw/cleaned_threads_v2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "deduplicated_threads.json")
MIN_EMAILS_AFTER_DEDUP = 3
MIN_QUALITY_SCORE = 60
MIN_YEAR = 1995
MAX_EMAILS_PER_THREAD = 30

def get_email_fingerprint(email):
    """
    Create a unique fingerprint for an email using multiple fields
    """
    # Get key fields
    sender = email.get('from', '').strip().lower()
    recipient = email.get('to', '').strip().lower()
    subject = email.get('subject', '').strip().lower()
    date = email.get('date', '').strip()
    
    # Use first 300 chars of body for fingerprint
    body = email.get('body', '')[:300].strip().lower()
    
    # Remove common variations
    body = body.replace('\n', ' ').replace('\r', ' ')
    body = ' '.join(body.split())  # Normalize whitespace
    
    # Create fingerprint string
    fingerprint_str = f"{sender}|{recipient}|{subject}|{date}|{body}"
    
    # Hash it for consistent comparison
    fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    return fingerprint

def get_email_timestamp(email):
    """
    Extract timestamp from email for sorting
    """
    date_str = email.get('date', '')
    try:
        # Parse ISO format timestamp
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return date_obj
    except:
        # Return a very old date if parsing fails
        return datetime(1970, 1, 1)

def remove_duplicates_from_thread(thread):
    """
    Remove duplicate emails from a thread while preserving chronological order
    Uses fingerprinting to detect exact duplicates
    """
    emails = thread.get('emails', [])
    
    # Sort by timestamp first
    emails.sort(key=get_email_timestamp)
    
    seen_fingerprints = set()
    unique_emails = []
    duplicates_removed = 0
    duplicate_details = []
    
    for i, email in enumerate(emails):
        fingerprint = get_email_fingerprint(email)
        
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            # Add timestamp info to email
            email['timestamp_parsed'] = get_email_timestamp(email).isoformat()
            unique_emails.append(email)
        else:
            duplicates_removed += 1
            # Track which emails were duplicates
            date_str = email.get('date', 'unknown')[:19]
            duplicate_details.append(f"Email {i+1}: {date_str}")
    
    return unique_emails, duplicates_removed, duplicate_details

def filter_by_date(emails, min_year=1995):
    """
    Remove emails with unrealistic dates (before min_year)
    """
    filtered_emails = []
    removed_count = 0
    bad_date_details = []
    
    for email in emails:
        timestamp = get_email_timestamp(email)
        
        if timestamp.year >= min_year:
            filtered_emails.append(email)
        else:
            removed_count += 1
            date_str = email.get('date', 'unknown')[:19]
            bad_date_details.append(f"Bad date: {date_str} (year: {timestamp.year})")
    
    return filtered_emails, removed_count, bad_date_details

def check_conversation_quality(emails):
    """
    Check if emails form a real back-and-forth conversation
    Returns True if it's a genuine conversation
    """
    if len(emails) < 3:
        return False
    
    # Get senders in order
    senders = [e.get('from', '') for e in emails]
    
    # Count alternations
    alternations = sum(1 for i in range(1, len(senders)) if senders[i] != senders[i-1])
    
    # Need at least 2 alternations for a conversation
    if alternations < 2:
        return False
    
    # Check unique senders
    unique_senders = len(set(senders))
    if unique_senders < 2:
        return False
    
    return True

def recalculate_quality_score(thread):
    """
    Recalculate quality score after deduplication
    """
    emails = thread['emails']
    if len(emails) < 3:
        return 0
    
    score = 0
    
    # 1. Sender alternation (40 points)
    senders = [e.get('from', '') for e in emails if e.get('from')]
    if len(senders) > 1:
        alternations = sum(1 for i in range(1, len(senders)) if senders[i] != senders[i-1])
        alternation_ratio = alternations / (len(senders) - 1) if len(senders) > 1 else 0
        alternation_score = min(40, alternation_ratio * 50)
        score += alternation_score
    
    # 2. Number of unique participants (30 points)
    all_participants = set()
    for email in emails:
        if email.get('from'):
            all_participants.add(email['from'])
        if email.get('to'):
            all_participants.add(email['to'])
    
    unique_participants = len(all_participants)
    participant_score = min(30, (unique_participants - 1) * 15)
    score += participant_score
    
    # 3. Thread length (30 points) - sweet spot is 5-15 emails
    num_emails = len(emails)
    if 5 <= num_emails <= 15:
        length_score = 30
    elif 3 <= num_emails < 5:
        length_score = 20
    elif 15 < num_emails <= 25:
        length_score = 25
    else:
        length_score = 15
    score += length_score
    
    return int(score)

def clean_threads(input_file, output_dir, output_file):
    """
    Main cleaning function with comprehensive deduplication
    """
    print("="*80)
    print("COMPREHENSIVE THREAD DEDUPLICATION V2")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load threads
    print(f"\n[1/6] Loading threads from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        threads = json.load(f)
    
    print(f"      Original threads: {len(threads):,}")
    
    # Statistics
    total_duplicates = 0
    total_bad_dates = 0
    threads_removed = 0
    low_quality_removed = 0
    
    # Process each thread
    print(f"\n[2/6] Removing duplicates and cleaning...")
    cleaned_threads = {}
    
    for thread_id, thread in threads.items():
        # Remove duplicates
        unique_emails, dups_removed, dup_details = remove_duplicates_from_thread(thread)
        total_duplicates += dups_removed
        
        # Filter by date
        valid_emails, bad_dates, bad_date_details = filter_by_date(unique_emails, MIN_YEAR)
        total_bad_dates += bad_dates
        
        # Check if thread still has enough emails
        if len(valid_emails) >= MIN_EMAILS_AFTER_DEDUP:
            # Check conversation quality
            if check_conversation_quality(valid_emails):
                # Update thread
                thread['emails'] = valid_emails
                thread['num_emails'] = len(valid_emails)
                thread['duplicates_removed'] = dups_removed
                thread['bad_dates_removed'] = bad_dates
                
                # Update date range
                timestamps = [get_email_timestamp(e) for e in valid_emails]
                thread['date_start'] = min(timestamps).isoformat()
                thread['date_end'] = max(timestamps).isoformat()
                
                # Recalculate quality score
                new_quality = recalculate_quality_score(thread)
                thread['quality_score'] = new_quality
                
                # Only keep if quality is still good and reasonable length
                if new_quality >= MIN_QUALITY_SCORE and len(valid_emails) <= MAX_EMAILS_PER_THREAD:
                    cleaned_threads[thread_id] = thread
                else:
                    low_quality_removed += 1
            else:
                threads_removed += 1
        else:
            threads_removed += 1
    
    print(f"      Duplicates removed: {total_duplicates:,}")
    print(f"      Bad dates removed: {total_bad_dates:,}")
    print(f"      Threads removed (too short/no conversation): {threads_removed:,}")
    print(f"      Threads removed (low quality): {low_quality_removed:,}")
    print(f"      Cleaned threads: {len(cleaned_threads):,}")
    
    # Sort by quality score
    print(f"\n[3/6] Sorting by quality score...")
    sorted_threads = dict(sorted(
        cleaned_threads.items(),
        key=lambda x: x[1]['quality_score'],
        reverse=True
    ))
    
    # Save cleaned threads
    print(f"\n[4/6] Saving cleaned threads...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_threads, f, indent=2, ensure_ascii=False)
    
    print(f"      Saved to: {output_file}")
    
    # Save top 200 threads separately
    print(f"\n[5/6] Saving top-quality subsets...")
    
    top_200_file = os.path.join(output_dir, "top_200_threads.json")
    top_200 = dict(list(sorted_threads.items())[:200])
    with open(top_200_file, 'w', encoding='utf-8') as f:
        json.dump(top_200, f, indent=2, ensure_ascii=False)
    print(f"      Top 200: {top_200_file}")
    
    # Save top 150 for dataset
    top_150_file = os.path.join(output_dir, "top_150_for_dataset.json")
    top_150 = dict(list(sorted_threads.items())[:150])
    with open(top_150_file, 'w', encoding='utf-8') as f:
        json.dump(top_150, f, indent=2, ensure_ascii=False)
    print(f"      Top 150 (for dataset): {top_150_file}")
    
    # Statistics
    print(f"\n[6/6] Final Statistics")
    print(f"\n{'='*80}")
    print("CLEANING COMPLETE")
    print(f"{'='*80}")
    print(f"Original threads: {len(threads):,}")
    print(f"After cleaning: {len(cleaned_threads):,}")
    print(f"Reduction: {len(threads) - len(cleaned_threads):,} threads ({((len(threads) - len(cleaned_threads))/len(threads)*100):.1f}%)")
    print(f"\nTotal duplicates removed: {total_duplicates:,}")
    print(f"Total bad dates removed: {total_bad_dates:,}")
    print(f"Threads removed (too short): {threads_removed:,}")
    print(f"Threads removed (low quality): {low_quality_removed:,}")
    
    # Thread length distribution
    print(f"\nThread length distribution (after cleaning):")
    length_dist = defaultdict(int)
    for thread in cleaned_threads.values():
        length = thread['num_emails']
        if length <= 10:
            length_dist[length] += 1
        elif length <= 20:
            length_dist['11-20'] += 1
        else:
            length_dist['21+'] += 1
    
    for length in sorted([k for k in length_dist.keys() if isinstance(k, int)]):
        print(f"  {length} emails: {length_dist[length]:,} threads")
    if '11-20' in length_dist:
        print(f"  11-20 emails: {length_dist['11-20']:,} threads")
    if '21+' in length_dist:
        print(f"  21+ emails: {length_dist['21+']:,} threads")
    
    # Quality score distribution
    print(f"\nQuality score distribution:")
    score_ranges = [
        (60, 70, '60-69'),
        (70, 80, '70-79'),
        (80, 90, '80-89'),
        (90, 101, '90-100')
    ]
    
    for low, high, label in score_ranges:
        count = sum(1 for t in cleaned_threads.values() if low <= t['quality_score'] < high)
        print(f"  {label}: {count:,} threads")
    
    # Top 10 threads
    print(f"\n{'='*80}")
    print("TOP 10 HIGHEST QUALITY THREADS (AFTER DEDUPLICATION)")
    print(f"{'='*80}")
    
    top_10 = list(sorted_threads.items())[:10]
    for i, (thread_id, thread) in enumerate(top_10, 1):
        subject = thread['subject'][:60]
        num_emails = thread['num_emails']
        quality = thread['quality_score']
        participants = thread.get('participants', [])
        dups_removed = thread.get('duplicates_removed', 0)
        
        date_start = thread['date_start'][:10]
        date_end = thread['date_end'][:10]
        
        print(f"\n{i}. Thread: {thread_id}")
        print(f"   Subject: {subject}...")
        print(f"   Emails: {num_emails} | Quality: {quality}/100")
        print(f"   Participants: {', '.join([p.split('@')[0] for p in participants])}")
        print(f"   Date range: {date_start} to {date_end}")
        if dups_removed > 0:
            print(f"   Duplicates removed: {dups_removed}")
    
    print(f"\n{'='*80}")
    print("READY FOR DATASET CREATION")
    print(f"{'='*80}")
    print(f"\nTop 150 threads saved to: {top_150_file}")
    print(f"These are ready to use as legitimate threads in your dataset!")
    print(f"\nNext step: Generate 200 attack threads with LLMs")
    
    return sorted_threads

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        print("Please run step3_extract_threads_IMPROVED.py first")
    else:
        cleaned = clean_threads(INPUT_FILE, OUTPUT_DIR, OUTPUT_FILE)