"""
Extract 12 thread-level features from email threads
Phase 2: Feature Engineering
"""

import json
import numpy as np
from datetime import datetime
from collections import Counter
import re

# Install required packages:
# pip install pandas numpy scikit-learn nltk textblob

def parse_date(date_str):
    """Parse email date string"""
    if not date_str or not isinstance(date_str, str):
        return None
    
    try:
        # Handle timezone offset format: 2001-05-23T07:14:00-07:00
        if 'T' in date_str:
            # Remove microseconds if present
            date_str = date_str.split('.')[0]
            
            # Handle timezone
            if '+' in date_str or date_str.count('-') > 2:
                # Has timezone - parse with fromisoformat
                return datetime.fromisoformat(date_str)
            else:
                # No timezone
                return datetime.fromisoformat(date_str)
        
        return None
    except:
        return None

def extract_temporal_features(thread):
    """
    Temporal Features:
    1. Response Time Variance
    2. Thread Length (days)
    3. Timestamp Anomalies
    """
    emails = thread['emails']
    dates = []
    
    for email in emails:
        date = parse_date(email.get('date', ''))
        if date:
            dates.append(date)
    
    if len(dates) < 2:
        return {'response_time_variance': 0, 'thread_length_days': 0, 'timestamp_anomalies': 0}
    
    # Sort dates
    dates.sort()
    
    # Response times (hours between emails)
    response_times = [(dates[i+1] - dates[i]).total_seconds() / 3600 for i in range(len(dates)-1)]
    
    # Response time variance
    response_time_variance = np.var(response_times) if len(response_times) > 1 else 0
    
    # Thread length (days)
    thread_length_days = (dates[-1] - dates[0]).total_seconds() / 86400
    
    # Timestamp anomalies (emails sent at unusual hours: 11pm-5am)
    anomalies = sum(1 for d in dates if d.hour >= 23 or d.hour <= 5)
    
    return {
        'response_time_variance': response_time_variance,
        'thread_length_days': thread_length_days,
        'timestamp_anomalies': anomalies / len(dates) if dates else 0
    }

def extract_linguistic_features(thread):
    """
    Linguistic Features:
    4. Tone Shift Detection
    5. Formality Variance
    6. Urgency Escalation
    """
    emails = thread['emails']
    
    # Urgency keywords
    urgency_words = [
        'urgent', 'asap', 'immediately', 'emergency', 'critical', 'now',
        'rush', 'hurry', 'quick', 'fast', 'deadline', 'crisis', 'catastrophic',
        'disaster', 'panic', 'desperate'
    ]
    
    urgency_scores = []
    formality_scores = []
    
    for email in emails:
        body = email.get('body', '').lower()
        
        # Urgency score (count of urgency words)
        urgency_count = sum(1 for word in urgency_words if word in body)
        urgency_scores.append(urgency_count)
        
        # Formality score (simple heuristic: formal words vs informal)
        formal_words = ['please', 'thank', 'regards', 'sincerely', 'respectfully']
        informal_words = ['hey', 'gonna', 'wanna', 'yeah', 'ok', 'btw']
        
        formal_count = sum(1 for word in formal_words if word in body)
        informal_count = sum(1 for word in informal_words if word in body)
        
        formality = formal_count - informal_count
        formality_scores.append(formality)
    
    # Tone shift: change in formality between first and last email
    tone_shift = abs(formality_scores[-1] - formality_scores[0]) if len(formality_scores) > 1 else 0
    
    # Formality variance
    formality_variance = np.var(formality_scores) if len(formality_scores) > 1 else 0
    
    # Urgency escalation: positive slope = escalating urgency
    if len(urgency_scores) > 1:
        urgency_escalation = urgency_scores[-1] - urgency_scores[0]
    else:
        urgency_escalation = 0
    
    return {
        'tone_shift': tone_shift,
        'formality_variance': formality_variance,
        'urgency_escalation': urgency_escalation
    }

def extract_coherence_features(thread):
    """
    Coherence Features:
    7. Reference Accuracy
    8. Entity Consistency
    9. Topic Coherence
    """
    emails = thread['emails']
    
    # Reference accuracy: presence of conversational references
    reference_words = ['as we discussed', 'as mentioned', 'per our', 'following up', 
                       'our call', 'our meeting', 'we talked', 'you said']
    
    reference_count = 0
    for email in emails:
        body = email.get('body', '').lower()
        reference_count += sum(1 for phrase in reference_words if phrase in body)
    
    reference_accuracy = reference_count / len(emails) if emails else 0
    
    # Entity consistency: count unique email addresses
    email_addresses = set()
    for email in emails:
        from_addr = email.get('from', '')
        to_addr = email.get('to', '')
        if from_addr:
            email_addresses.add(from_addr.lower())
        if to_addr:
            email_addresses.add(to_addr.lower())
    
    entity_consistency = 1.0 / len(email_addresses) if email_addresses else 1.0
    
    # Topic coherence: subject consistency (same base subject)
    subjects = [email.get('subject', '').lower() for email in emails]
    base_subjects = [re.sub(r'^(re:|fwd:)\s*', '', s).strip() for s in subjects]
    unique_subjects = len(set(base_subjects))
    
    topic_coherence = 1.0 / unique_subjects if unique_subjects > 0 else 1.0
    
    return {
        'reference_accuracy': reference_accuracy,
        'entity_consistency': entity_consistency,
        'topic_coherence': topic_coherence
    }

def extract_context_features(thread):
    """
    Context Features:
    10. Fabricated History Score
    11. Relationship Velocity
    12. Cross-Reference Count
    """
    emails = thread['emails']
    
    # Fabricated history: references to non-existent prior communication
    fabrication_phrases = [
        'as we discussed', 'per our conversation', 'following our meeting',
        'as mentioned in our call', 'from our discussion', 'per our agreement',
        'as we agreed', 'from yesterday', 'last week', 'our session'
    ]
    
    fabrication_count = 0
    for email in emails:
        body = email.get('body', '').lower()
        fabrication_count += sum(1 for phrase in fabrication_phrases if phrase in body)
    
    fabricated_history_score = fabrication_count / len(emails) if emails else 0
    
    # Relationship velocity: formality change rate
    formality_changes = 0
    for i in range(1, len(emails)):
        prev_body = emails[i-1].get('body', '').lower()
        curr_body = emails[i].get('body', '').lower()
        
        # Check for formality shift (dear -> hey, regards -> none)
        if ('dear' in prev_body and 'hey' in curr_body) or \
           ('regards' in prev_body and 'regards' not in curr_body):
            formality_changes += 1
    
    relationship_velocity = formality_changes / (len(emails) - 1) if len(emails) > 1 else 0
    
    # Cross-reference count: explicit references to other emails/messages
    cross_ref_words = ['forward', 'fwd', 'attachment', 'attached', 'cc', 'bcc', 'replied']
    cross_ref_count = 0
    
    for email in emails:
        body = email.get('body', '').lower()
        cross_ref_count += sum(1 for word in cross_ref_words if word in body)
    
    cross_reference_count = cross_ref_count / len(emails) if emails else 0
    
    return {
        'fabricated_history_score': fabricated_history_score,
        'relationship_velocity': relationship_velocity,
        'cross_reference_count': cross_reference_count
    }

def extract_all_features(thread):
    """Extract all 12 features for a thread"""
    features = {}
    
    # Basic features
    features['thread_id'] = thread['thread_id']
    features['label'] = thread['label']
    features['attack_type'] = thread['attack_type']
    features['num_emails'] = thread['num_emails']
    
    # Temporal features
    features.update(extract_temporal_features(thread))
    
    # Linguistic features
    features.update(extract_linguistic_features(thread))
    
    # Coherence features
    features.update(extract_coherence_features(thread))
    
    # Context features
    features.update(extract_context_features(thread))
    
    return features

def main():
    print("="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    # Load datasets
    datasets = {
        'train': '../../data/splits/train_dataset.json',
        'val': '../../data/splits/val_dataset.json',
        'test': '../../data/splits/test_dataset.json'
    }
    
    for split_name, filepath in datasets.items():
        print(f"\n📊 Processing {split_name} set...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            threads = json.load(f)
        
        # Extract features
        features_list = []
        for i, thread in enumerate(threads):
            features = extract_all_features(thread)
            features_list.append(features)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(threads)} threads")
        
        # Save features
        output_file = f'../../data/features/{split_name}_features.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(features_list, f, indent=2)
        
        print(f"  ✓ Saved: {output_file} ({len(features_list)} threads)")
    
    print("\n" + "="*60)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print("\nNext step: Train models with extracted features")

if __name__ == '__main__':
    main()