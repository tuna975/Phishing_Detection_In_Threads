"""
Step 3 IMPROVED: Thread Extractor with Participant Grouping
Groups emails by BOTH subject AND participants for accurate threading
"""

import json
from collections import defaultdict
from datetime import datetime
import os

# Configuration
INPUT_JSON = "../../data/raw/enron_prepared/enron_emails_for_threads.json"
OUTPUT_DIR = "../../data/raw/extracted_threads_v2"
MIN_EMAILS_PER_THREAD = 3
MIN_EXCHANGES = 2  # Minimum back-and-forth exchanges

def normalize_email(email):
    """Normalize email address"""
    if not email:
        return None
    return email.strip().lower()

def get_participant_set(email):
    """Get all participants (sender + recipient) from an email"""
    participants = set()
    
    sender = normalize_email(email.get('from'))
    recipient = normalize_email(email.get('to'))
    
    if sender:
        participants.add(sender)
    if recipient:
        participants.add(recipient)
    
    return participants

def get_thread_key(subject_base, participants):
    """
    Create a unique key for a thread based on subject + participants
    """
    # Sort participants for consistent key
    participant_list = sorted(list(participants))
    participant_key = "|".join(participant_list)
    
    # Combine subject and participants
    thread_key = f"{subject_base}:::{participant_key}"
    
    return thread_key

def find_threads_by_subject_and_participants(emails):
    """
    Group emails into threads based on BOTH subject AND participants
    """
    print(f"\n[FINDING THREADS] Grouping by subject AND participants...")
    
    # Group by thread key (subject + participants)
    thread_groups = defaultdict(list)
    
    for email in emails:
        subject_base = email.get('subject_base', '').lower().strip()
        
        if not subject_base or subject_base == 'no_subject':
            continue
        
        # Get participants
        participants = get_participant_set(email)
        
        if len(participants) < 2:  # Need at least sender and recipient
            continue
        
        # Create thread key
        thread_key = get_thread_key(subject_base, participants)
        
        thread_groups[thread_key].append(email)
    
    # Convert to thread format
    threads = {}
    thread_id = 1
    
    for thread_key, email_list in thread_groups.items():
        # Only keep threads with minimum emails
        if len(email_list) < MIN_EMAILS_PER_THREAD:
            continue
        
        # Sort by date
        email_list.sort(key=lambda x: x.get('date', ''))
        
        # Extract subject and participants from key
        subject_base = thread_key.split(':::')[0]
        participant_str = thread_key.split(':::')[1]
        participants = participant_str.split('|')
        
        # Check for actual back-and-forth (different senders)
        senders = [e.get('from') for e in email_list if e.get('from')]
        unique_senders = len(set(senders))
        
        # Require at least 2 different senders for a conversation
        if unique_senders >= 2:
            threads[f"thread_{thread_id:04d}"] = {
                'thread_id': f"thread_{thread_id:04d}",
                'subject': subject_base,
                'num_emails': len(email_list),
                'participants': participants,
                'num_participants': len(participants),
                'date_start': email_list[0].get('date', ''),
                'date_end': email_list[-1].get('date', ''),
                'emails': email_list
            }
            thread_id += 1
    
    print(f"[FOUND] {len(threads)} conversation threads")
    return threads

def analyze_thread_quality(thread):
    """
    Analyze thread quality based on conversation dynamics
    Returns quality score (0-100)
    """
    emails = thread['emails']
    if len(emails) < MIN_EMAILS_PER_THREAD:
        return 0
    
    score = 0
    
    # 1. Sender alternation (40 points max)
    senders = [e.get('from') for e in emails if e.get('from')]
    if len(senders) > 1:
        alternations = sum(1 for i in range(1, len(senders)) if senders[i] != senders[i-1])
        alternation_ratio = alternations / (len(senders) - 1) if len(senders) > 1 else 0
        alternation_score = min(40, alternation_ratio * 50)  # More weight to alternation
        score += alternation_score
    
    # 2. Number of unique participants (30 points max)
    num_participants = thread['num_participants']
    participant_score = min(30, (num_participants - 1) * 15)  # 2 people = 15, 3 = 30
    score += participant_score
    
    # 3. Thread length (30 points max)
    # Sweet spot: 5-15 emails
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

def display_thread(thread):
    """Display a single thread"""
    print(f"\n{'='*80}")
    print(f"Thread ID: {thread['thread_id']}")
    print(f"Subject: {thread['subject']}")
    print(f"Emails: {thread['num_emails']} | Participants: {thread['num_participants']}")
    print(f"Participants: {', '.join([p.split('@')[0] for p in thread['participants']])}")
    print(f"Date Range: {thread['date_start'][:10]} to {thread['date_end'][:10]}")
    print(f"Quality: {thread.get('quality_score', 0)}/100")
    print(f"{'='*80}")
    
    for i, email in enumerate(thread['emails'], 1):
        sender = email['from'].split('@')[0] if '@' in email['from'] else email['from']
        recipient = email['to'].split('@')[0] if '@' in email['to'] else email['to']
        date = email['date'][:10] if email.get('date') else 'unknown'
        
        print(f"\n[Email {i}] {date} | {sender} → {recipient}")
        
        body = email.get('body', '')
        if body:
            preview = body[:150].replace('\n', ' ')
            print(f"Body: {preview}...")
        print("-" * 80)

def extract_and_save_threads(input_json, output_dir, quality_threshold=50):
    """
    Main function to extract threads with participant grouping
    """
    print("="*80)
    print("IMPROVED THREAD EXTRACTION (Subject + Participants)")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load emails
    print(f"\n[1/5] Loading emails from: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        emails = json.load(f)
    print(f"      Total emails loaded: {len(emails):,}")
    
    # Find threads
    print(f"\n[2/5] Finding conversation threads...")
    threads = find_threads_by_subject_and_participants(emails)
    print(f"      Threads found: {len(threads):,}")
    
    # Analyze quality
    print(f"\n[3/5] Analyzing thread quality...")
    for thread_id, thread in threads.items():
        thread['quality_score'] = analyze_thread_quality(thread)
    
    # Filter by quality
    high_quality_threads = {
        tid: t for tid, t in threads.items() 
        if t['quality_score'] >= quality_threshold
    }
    
    print(f"      High-quality threads (score >= {quality_threshold}): {len(high_quality_threads):,}")
    
    # Save threads
    print(f"\n[4/5] Saving threads to JSON...")
    
    # Save all threads
    all_threads_file = os.path.join(output_dir, "all_threads.json")
    with open(all_threads_file, 'w', encoding='utf-8') as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"      Saved all threads to: {all_threads_file}")
    
    # Save high-quality threads
    hq_threads_file = os.path.join(output_dir, "high_quality_threads.json")
    with open(hq_threads_file, 'w', encoding='utf-8') as f:
        json.dump(high_quality_threads, f, indent=2, ensure_ascii=False)
    print(f"      Saved high-quality threads to: {hq_threads_file}")
    
    # Save individual thread files
    individual_dir = os.path.join(output_dir, "individual_threads")
    os.makedirs(individual_dir, exist_ok=True)
    
    for thread_id, thread in high_quality_threads.items():
        thread_file = os.path.join(individual_dir, f"{thread_id}.json")
        with open(thread_file, 'w', encoding='utf-8') as f:
            json.dump(thread, f, indent=2, ensure_ascii=False)
    
    print(f"      Saved individual threads to: {individual_dir}")
    
    # Statistics
    print(f"\n[5/5] Statistics")
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total threads found: {len(threads):,}")
    print(f"High-quality threads: {len(high_quality_threads):,}")
    
    # Thread length distribution
    print(f"\nThread length distribution:")
    length_dist = defaultdict(int)
    for thread in high_quality_threads.values():
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
    
    # Participant distribution
    print(f"\nParticipant distribution:")
    participant_dist = defaultdict(int)
    for thread in high_quality_threads.values():
        num_p = thread['num_participants']
        participant_dist[num_p] += 1
    
    for num_p in sorted(participant_dist.keys()):
        print(f"  {num_p} participants: {participant_dist[num_p]:,} threads")
    
    # Quality score distribution
    print(f"\nQuality score distribution:")
    score_ranges = [(50, 60), (60, 70), (70, 80), (80, 100)]
    for low, high in score_ranges:
        count = sum(1 for t in high_quality_threads.values() if low <= t['quality_score'] < high)
        print(f"  {low}-{high}: {count:,} threads")
    
    # Show top 5 threads
    print(f"\n{'='*80}")
    print("TOP 5 HIGHEST QUALITY THREADS")
    print(f"{'='*80}")
    
    sorted_threads = sorted(
        high_quality_threads.values(),
        key=lambda x: x['quality_score'],
        reverse=True
    )[:5]
    
    for thread in sorted_threads:
        print(f"\nThread: {thread['thread_id']}")
        print(f"Subject: {thread['subject'][:60]}...")
        print(f"Emails: {thread['num_emails']} | Quality: {thread['quality_score']}/100")
        print(f"Participants ({thread['num_participants']}): {', '.join([p.split('@')[0] for p in thread['participants'][:5]])}...")
    
    print(f"\n{'='*80}")
    
    return high_quality_threads

if __name__ == "__main__":
    if not os.path.exists(INPUT_JSON):
        print(f"[ERROR] Input file not found: {INPUT_JSON}")
        print("Please run step2_prepare_enron_data.py first")
    else:
        # Extract threads
        threads = extract_and_save_threads(INPUT_JSON, OUTPUT_DIR, quality_threshold=50)
        
        # Optional: Show sample threads
        print(f"\n\nWould you like to see 3 sample threads? (y/n): ", end='')
        response = input().strip().lower()
        if response == 'y':
            sample_threads = list(threads.values())[:3]
            for thread in sample_threads:
                display_thread(thread)