
import json
import numpy as np
from datetime import datetime
import re
from collections import Counter
from dateutil import parser as date_parser



try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import spacy
    from textblob import TextBlob
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some libraries not installed: {e}")
    print("Script will use fallback methods for missing libraries")
    LIBRARIES_AVAILABLE = False

# Initialize NLP models (global to avoid reloading)
if LIBRARIES_AVAILABLE:
    try:
        vader_analyzer = SentimentIntensityAnalyzer()
        nlp_spacy = spacy.load("en_core_web_sm")
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ All NLP libraries loaded successfully\n")
    except Exception as e:
        print(f"⚠️  Warning loading models: {e}")
        print("Using fallback methods\n")
        LIBRARIES_AVAILABLE = False


def parse_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        dt = date_parser.parse(date_str)
        # Convert to UTC if timezone-aware, strip timezone info
        if dt.tzinfo:
            dt = dt.utctimetuple()
            return datetime(dt.tm_year, dt.tm_mon, dt.tm_mday, 
                          dt.tm_hour, dt.tm_min, dt.tm_sec)
        return dt
    except Exception:
        print(f"    ⚠️  Could not parse date: {date_str}")
        return None
# ============================================================================
# TEMPORAL FEATURES (Citations: Verma 2012, Repke & Krestel 2018, Das 2019)
# ============================================================================

def extract_temporal_features(thread):
    """
    Extract temporal features from email thread
    
    Features:
    1. Response Time Variance (Verma et al., 2012)
       - Variance in time gaps between consecutive emails
    
    2. Thread Length in Days (Repke & Krestel, 2018)  
       - Total duration from first to last email
    
    3. Timestamp Anomalies (Das et al., 2019)
       - Proportion of emails sent during unusual hours
    """
    emails = thread['emails']
    dates = []
    
    # Parse all dates
    for email in emails:
        date = parse_date(email.get('date', ''))
        if date:
            dates.append(date)
    
    # Need at least 2 emails for temporal analysis
    if len(dates) < 2:
        return {
            'response_time_variance': 0, 
            'thread_length_days': 0, 
            'timestamp_anomalies': 0
        }
    
    # Sort dates chronologically
    dates.sort()
    
    # 1. Response Time Variance
    # Calculate time gaps in hours
    response_times = [
        (dates[i+1] - dates[i]).total_seconds() / 3600 
        for i in range(len(dates)-1)
    ]
    response_time_variance = np.var(response_times) if len(response_times) > 1 else 0
    
    # 2. Thread Length (days)
    thread_length_days = (dates[-1] - dates[0]).total_seconds() / 86400
    
    # 3. Timestamp Anomalies
    # Unusual hours: before 6am or after 10pm (Das et al., 2019)
    anomaly_count = sum(1 for d in dates if d.hour < 6 or d.hour >= 22)
    timestamp_anomalies = anomaly_count / len(dates)
    
    return {
        'response_time_variance': float(response_time_variance),
        'thread_length_days': float(thread_length_days),
        'timestamp_anomalies': float(timestamp_anomalies)
    }


# ============================================================================
# LINGUISTIC FEATURES (Citations: Hutto 2014, Verma 2012, Das 2019)
# ============================================================================

def calculate_vader_sentiment(text):
    """
    Calculate sentiment using VADER (Hutto & Gilbert, 2014)
    Returns compound score: -1 (negative) to +1 (positive)
    """
    if LIBRARIES_AVAILABLE:
        try:
            scores = vader_analyzer.polarity_scores(text)
            return scores['compound']
        except:
            pass
    
    # Fallback: simple positive/negative word counting
    positive_words = ['good', 'great', 'excellent', 'thank', 'please', 'appreciate']
    negative_words = ['bad', 'urgent', 'immediate', 'problem', 'issue', 'error']
    
    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total


def calculate_formality_score(text):
    """
    Calculate formality using TextBlob-style analysis (Loria, 2018)
    Higher score = more formal
    """
    if LIBRARIES_AVAILABLE:
        try:
            blob = TextBlob(text)
            
            formality = 0
            
            # Formal greetings and closings
            formal_markers = [
                'dear', 'respectfully', 'sincerely', 'regards', 'best regards',
                'cordially', 'thank you', 'please'
            ]
            for marker in formal_markers:
                if marker in text.lower():
                    formality += 1
            
            # Complete sentences (TextBlob sentence detection)
            sentences = blob.sentences
            if len(sentences) > 0:
                complete_sentences = sum(1 for s in sentences if str(s).strip().endswith('.'))
                formality += complete_sentences / len(sentences)
            
            # Average word length (formal writing uses longer words)
            words = blob.words
            if len(words) > 0:
                avg_word_length = sum(len(w) for w in words) / len(words)
                formality += avg_word_length / 10
            
            return formality
        except:
            pass
    
    # Fallback formality calculation
    text_lower = text.lower()
    
    formality = 0
    
    # Formal markers
    formal_words = ['dear', 'respectfully', 'sincerely', 'regards', 'please', 'thank you']
    formality += sum(1 for w in formal_words if w in text_lower)
    
    # Informal markers (reduce formality)
    informal_words = ['hey', 'gonna', 'wanna', 'yeah', 'ok', 'btw']
    formality -= sum(1 for w in informal_words if w in text_lower)
    
    return max(0, formality)  # Don't go negative


def extract_linguistic_features(thread):
    """
    Extract linguistic features using NLP libraries
    
    Features:
    4. Tone Shift (Hutto & Gilbert, 2014 - VADER)
       - Change in sentiment/tone across thread using VADER sentiment analysis
    
    5. Formality Variance (Verma et al., 2012; Loria, 2018 - TextBlob)
       - Variance in writing formality across emails
    
    6. Urgency Escalation (Das et al., 2019)
       - Increase in urgency keywords over thread
    """
    emails = thread['emails']
    
    sentiment_scores = []
    formality_scores = []
    urgency_scores = []
    
    # Urgency keywords from phishing research (Das et al., 2019)
    urgency_keywords = [
        'urgent', 'asap', 'immediately', 'emergency', 'critical', 'now',
        'rush', 'hurry', 'quick', 'fast', 'deadline', 'today', 'tonight',
        'expires', 'expiring', 'limited time', 'act now'
    ]
    
    for email in emails:
        body = email.get('body', '')
        body_lower = body.lower()
        
        # 1. Sentiment (VADER)
        sentiment = calculate_vader_sentiment(body)
        sentiment_scores.append(sentiment)
        
        # 2. Formality (TextBlob-style)
        formality = calculate_formality_score(body)
        formality_scores.append(formality)
        
        # 3. Urgency count
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in body_lower)
        urgency_scores.append(urgency_count)
    
    # Calculate features
    
    # Tone Shift: variance in sentiment across thread
    tone_shift = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
    
    # Formality Variance
    formality_variance = np.var(formality_scores) if len(formality_scores) > 1 else 0
    
    # Urgency Escalation: difference between last and first
    if len(urgency_scores) > 1:
        urgency_escalation = urgency_scores[-1] - urgency_scores[0]
    else:
        urgency_escalation = 0
    
    return {
        'tone_shift': float(tone_shift),
        'formality_variance': float(formality_variance),
        'urgency_escalation': float(urgency_escalation)
    }


# ============================================================================
# COHERENCE FEATURES (Citations: Honnibal 2017, Reimers 2019, Verma 2012)
# ============================================================================

def extract_entities_spacy(text):
    """
    Extract named entities using spaCy (Honnibal & Montani, 2017)
    Returns set of entity texts (PERSON, ORG, GPE)
    """
    if LIBRARIES_AVAILABLE:
        try:
            doc = nlp_spacy(text)
            entities = set([
                ent.text.lower() 
                for ent in doc.ents 
                if ent.label_ in ['PERSON', 'ORG', 'GPE']
            ])
            return entities
        except:
            pass
    
    # Fallback: simple capitalized word extraction
    words = text.split()
    entities = set([
        word.lower() 
        for word in words 
        if word and word[0].isupper() and len(word) > 2
    ])
    return entities


def calculate_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity using BERT (Reimers & Gurevych, 2019)
    Returns similarity score 0-1
    """
    if LIBRARIES_AVAILABLE:
        try:
            embeddings = bert_model.encode([text1, text2])
            similarity = cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )[0][0]
            return similarity
        except:
            pass
    
    # Fallback: simple word overlap
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    union = len(words1 | words2)
    
    return overlap / union if union > 0 else 0.0


def extract_coherence_features(thread):
    """
    Extract coherence features using NLP
    
    Features:
    7. Reference Accuracy (Verma et al., 2012)
       - Presence of conversational references to prior communication
    
    8. Entity Consistency (Honnibal & Montani, 2017 - spaCy NER)
       - Consistency of named entities across thread
    
    9. Topic Coherence (Reimers & Gurevych, 2019 - BERT)
       - Semantic similarity between consecutive emails
    """
    emails = thread['emails']
    
    # 1. Reference Accuracy
    # Phrases that reference prior communication (Verma et al., 2012)
    reference_phrases = [
        'as we discussed', 'as mentioned', 'per our', 'following up',
        'our call', 'our meeting', 'we talked', 'you said', 'you mentioned',
        'previously', 'earlier', 'as noted'
    ]
    
    reference_count = 0
    for email in emails:
        body_lower = email.get('body', '').lower()
        reference_count += sum(1 for phrase in reference_phrases if phrase in body_lower)
    
    reference_accuracy = reference_count / len(emails) if emails else 0
    
    # 2. Entity Consistency (spaCy NER)
    all_entities = []
    for email in emails:
        body = email.get('body', '')
        entities = extract_entities_spacy(body)
        all_entities.append(entities)
    
    # Calculate entity overlap between consecutive emails
    if len(all_entities) < 2:
        entity_consistency = 1.0
    else:
        overlaps = []
        for i in range(len(all_entities) - 1):
            entities1 = all_entities[i]
            entities2 = all_entities[i + 1]
            
            if entities1 and entities2:
                overlap = len(entities1 & entities2) / max(len(entities1), len(entities2))
                overlaps.append(overlap)
        
        entity_consistency = np.mean(overlaps) if overlaps else 0.5
    
    # 3. Topic Coherence (BERT semantic similarity)
    if len(emails) < 2:
        topic_coherence = 1.0
    else:
        similarities = []
        for i in range(len(emails) - 1):
            text1 = emails[i].get('body', '')
            text2 = emails[i + 1].get('body', '')
            
            if text1 and text2:
                sim = calculate_semantic_similarity(text1, text2)
                similarities.append(sim)
        
        topic_coherence = np.mean(similarities) if similarities else 0.5
    
    return {
        'reference_accuracy': float(reference_accuracy),
        'entity_consistency': float(entity_consistency),
        'topic_coherence': float(topic_coherence)
    }


# ============================================================================
# CONTEXT-AWARE FEATURES (Citation: Hasan et al., 2025)
# ============================================================================

def extract_context_features(thread):
    """
    Extract context-aware features for detecting fabricated context
    
    Features based on Hasan et al. (2025) - LLM-PEA Framework
    
    10. Fabricated History Score
        - References to non-existent prior meetings/calls
    
    11. Relationship Velocity  
        - Speed of formality decrease / trust building
    
    12. Cross-Reference Count
        - Explicit references to other communications
    """
    emails = thread['emails']
    
    # 1. Fabricated History Score (Hasan et al., 2025)
    # Phrases claiming prior interaction that may be fabricated
    fabrication_indicators = [
        'as we discussed', 'per our conversation', 'following our meeting',
        'as mentioned in our call', 'from our discussion', 'per our agreement',
        'as we agreed', 'from yesterday', 'last week', 'our session',
        'during our call', 'in our meeting', 'when we talked', 'our chat'
    ]
    
    fabrication_count = 0
    for email in emails:
        body_lower = email.get('body', '').lower()
        fabrication_count += sum(1 for phrase in fabrication_indicators if phrase in body_lower)
    
    fabricated_history_score = fabrication_count / len(emails) if emails else 0
    
    # 2. Relationship Velocity (Hasan et al., 2025)
    # Rapid shift from formal to informal = social engineering
    formality_shifts = 0
    
    for i in range(1, len(emails)):
        prev_body = emails[i-1].get('body', '').lower()
        curr_body = emails[i].get('body', '').lower()
        
        # Detect formality drops
        formal_indicators = ['dear', 'sincerely', 'regards', 'respectfully']
        informal_indicators = ['hey', 'hi there', 'thanks!', 'cheers']
        
        prev_formal = any(ind in prev_body for ind in formal_indicators)
        curr_informal = any(ind in curr_body for ind in informal_indicators)
        
        if prev_formal and curr_informal:
            formality_shifts += 1
    
    relationship_velocity = formality_shifts / (len(emails) - 1) if len(emails) > 1 else 0
    
    # 3. Cross-Reference Count (Verma et al., 2012)
    # References to other messages/attachments
    cross_ref_keywords = [
        'forward', 'fwd', 'attachment', 'attached', 'file', 'document',
        'cc', 'bcc', 'replied', 'sent', 'email', 'message'
    ]
    
    cross_ref_count = 0
    for email in emails:
        body_lower = email.get('body', '').lower()
        cross_ref_count += sum(1 for keyword in cross_ref_keywords if keyword in body_lower)
    
    cross_reference_count = cross_ref_count / len(emails) if emails else 0
    
    return {
        'fabricated_history_score': float(fabricated_history_score),
        'relationship_velocity': float(relationship_velocity),
        'cross_reference_count': float(cross_reference_count)
    }


# ============================================================================
# MAIN FEATURE EXTRACTION
# ============================================================================

def extract_all_features(thread):
    """
    Extract all 12 thread-level features with proper academic citations
    
    Returns dictionary with:
    - Basic metadata (thread_id, label, attack_type, num_emails)
    - 3 temporal features
    - 3 linguistic features  
    - 3 coherence features
    - 3 context-aware features
    """
    features = {}
    
    # Basic metadata
    features['thread_id'] = thread['thread_id']
    features['label'] = thread['label']
    features['attack_type'] = thread.get('attack_type', 'unknown')
    features['num_emails'] = thread['num_emails']
    
    # Extract feature groups
    features.update(extract_temporal_features(thread))
    features.update(extract_linguistic_features(thread))
    features.update(extract_coherence_features(thread))
    features.update(extract_context_features(thread))
    
    return features


def main():
    """
    Main feature extraction pipeline
    Processes train, validation, and test datasets
    """
    print("="*70)
    print("THREAD-LEVEL FEATURE EXTRACTION FOR PHISHING DETECTION")
    print("="*70)
    print("\n📚 Using cited NLP libraries:")
    print("  • VADER (Hutto & Gilbert, 2014) - Sentiment analysis")
    print("  • spaCy (Honnibal & Montani, 2017) - Named entity recognition")
    print("  • TextBlob (Loria, 2018) - Formality analysis")
    print("  • BERT (Reimers & Gurevych, 2019) - Semantic similarity")
    print("\n📊 Extracting 12 features based on research:")
    print("  • Temporal (3): Verma 2012, Repke & Krestel 2018, Das 2019")
    print("  • Linguistic (3): Hutto 2014, Verma 2012, Das 2019")
    print("  • Coherence (3): Honnibal 2017, Reimers 2019, Verma 2012")
    print("  • Context-aware (3): Hasan et al. 2025")
    print("="*70)
    print()
    
    # Dataset files
    datasets = {
        'train': '../../data/splits/train_dataset.json',
        'val': '../../data/splits/val_dataset.json',
        'test': '../../data/splits/test_dataset.json'
    }
    
    # Process each dataset
    for split_name, filepath in datasets.items():
        print(f"📂 Processing {split_name.upper()} set...")
        print(f"   Loading: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                threads = json.load(f)
            
            print(f"   Found {len(threads)} threads")
            
            # Extract features for all threads
            features_list = []
            for i, thread in enumerate(threads):
                features = extract_all_features(thread)
                features_list.append(features)
                
                # Progress update every 50 threads
                if (i + 1) % 50 == 0:
                    print(f"   Progress: {i+1}/{len(threads)} threads processed")
            
            # Save features
            output_file = f'../../data/features/{split_name}_features.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(features_list, f, indent=2)
            
            print(f"   ✅ Saved: {output_file}")
            print(f"   ✅ Features extracted: {len(features_list)} threads")
            print()
            
        except FileNotFoundError:
            print(f"   ⚠️  File not found: {filepath}")
            print(f"   Skipping {split_name} set")
            print()
        except Exception as e:
            print(f"   ❌ Error processing {split_name}: {e}")
            print()
    
    print("="*70)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("="*70)
    print("\n📊 Feature Summary:")
    print("   • Total features per thread: 16 (4 metadata + 12 extracted)")
    print("   • All features have academic citations")
    print("   • Using state-of-the-art NLP libraries")
    print("\n📌 Next steps:")
    print("   1. Verify temporal features are non-zero for legitimate threads")
    print("   2. Train baseline Random Forest model")
    print("   3. Compare with thread-aware BERT model")
    print()


if __name__ == '__main__':
    main()