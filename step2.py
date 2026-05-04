"""
Step 2 UPDATED: Prepare Enron Fraud Dataset for Thread Extraction
Specifically designed for the Enron fraud dataset structure
"""

import pandas as pd
import json
import re
from datetime import datetime
import os

# Configuration
INPUT_CSV = "enron_data_fraud_labeled.csv"  # Update this path
OUTPUT_DIR = "enron_prepared"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "enron_emails_for_threads.json")

def clean_email_address(email_str):
    """Extract email address from various formats"""
    if pd.isna(email_str) or email_str == '':
        return None
    
    # Remove extra spaces
    email_str = str(email_str).strip()
    
    # Extract email from "Name <email@domain.com>" format
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_str)
    if match:
        return match.group(0).lower()
    
    return email_str.lower() if '@' in email_str else None

def extract_subject_base(subject):
    """Remove Re:, Fw:, Fwd: prefixes to group threads"""
    if pd.isna(subject) or subject == '':
        return "no_subject"
    
    subject = str(subject).strip()
    # Remove Re:, RE:, Fw:, FW:, Fwd:, FWD: prefixes (multiple times)
    for _ in range(3):  # Remove up to 3 levels of Re:/Fw:
        subject = re.sub(r'^(re|fw|fwd):\s*', '', subject, flags=re.IGNORECASE)
    
    return subject.strip() if subject.strip() else "no_subject"

def prepare_for_thread_extraction(input_csv, output_json):
    """
    Prepare Enron fraud emails for thread extraction
    Filter for non-fraud internal emails only
    """
    
    print("="*80)
    print("PREPARING ENRON FRAUD DATASET FOR THREAD EXTRACTION")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print(f"\n[1/7] Loading dataset from: {input_csv}")
    # Use low_memory=False to avoid dtype warnings
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"      Total emails: {len(df):,}")
    
    # Show label distribution
    print(f"\n[2/7] Label distribution:")
    print(df['Label'].value_counts())
    print(f"      0 = Legitimate")
    print(f"      1 = Fraud/Phishing")
    
    # Filter for legitimate emails only (Label = 0)
    print(f"\n[3/7] Filtering for legitimate (non-fraud) emails...")
    df = df[df['Label'] == 0]
    print(f"      Legitimate emails: {len(df):,}")
    
    # Filter for internal @enron.com emails only
    print(f"\n[4/7] Filtering for internal @enron.com conversations...")
    
    df['from_clean'] = df['From'].apply(clean_email_address)
    df['to_clean'] = df['To'].apply(clean_email_address)
    
    # Keep only @enron.com conversations
    before_count = len(df)
    df = df[
        (df['from_clean'].notna()) &
        (df['to_clean'].notna()) &
        (df['from_clean'].str.contains('@enron.com', na=False)) &
        (df['to_clean'].str.contains('@enron.com', na=False))
    ]
    print(f"      Internal @enron.com emails: {len(df):,} (filtered out {before_count - len(df):,})")
    
    # Remove emails without subjects (can't thread them)
    print(f"\n[5/7] Filtering emails with subjects...")
    df = df[df['Subject'].notna()]
    df = df[df['Subject'].str.strip() != '']
    print(f"      Emails with subjects: {len(df):,}")
    
    # Extract subject base for thread grouping
    print(f"\n[6/7] Processing subjects for thread grouping...")
    df['subject_base'] = df['Subject'].apply(extract_subject_base)
    
    # Remove "no_subject" emails
    df = df[df['subject_base'] != 'no_subject']
    print(f"      Valid subject emails: {len(df):,}")
    
    # Process dates
    df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    print(f"      Emails with valid dates: {len(df):,}")
    
    # Convert to JSON format for thread parser
    print(f"\n[7/7] Converting to JSON format...")
    
    emails_list = []
    for idx, row in df.iterrows():
        email_dict = {
            'message_id': str(row['Message-ID']),
            'from': row['from_clean'],
            'to': row['to_clean'],
            'subject': str(row['Subject']),
            'subject_base': row['subject_base'],
            'date': row['timestamp'].isoformat(),
            'body': str(row.get('Body', ''))
        }
        emails_list.append(email_dict)
    
    # Save to JSON
    print(f"      Saving to JSON...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(emails_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Saved {len(emails_list):,} emails to: {output_json}")
    
    # Statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    print(f"Total legitimate internal emails: {len(emails_list):,}")
    print(f"Unique senders: {df['from_clean'].nunique():,}")
    print(f"Unique recipients: {df['to_clean'].nunique():,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Top conversation subjects
    print(f"\nTop 15 most common conversation subjects:")
    subject_counts = df['subject_base'].value_counts().head(15)
    for i, (subject, count) in enumerate(subject_counts.items(), 1):
        truncated_subject = subject[:70] + "..." if len(subject) > 70 else subject
        print(f"  {i:2d}. [{count:4d} emails] {truncated_subject}")
    
    # Sender statistics
    print(f"\nTop 10 most active senders:")
    sender_counts = df['from_clean'].value_counts().head(10)
    for i, (sender, count) in enumerate(sender_counts.items(), 1):
        print(f"  {i:2d}. {sender:40s} - {count:4d} emails")
    
    print(f"\n{'='*80}")
    print("PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext step: Run step3_extract_threads.py")
    print(f"File location: {output_json}")
    
    return emails_list

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        print("Please update INPUT_CSV path at the top of this script")
        print(f"Current working directory: {os.getcwd()}")
    else:
        prepare_for_thread_extraction(INPUT_CSV, OUTPUT_JSON)
