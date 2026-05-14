"""
Model Comparison: Claude vs ChatGPT vs Grok
Evaluates phishing email generation quality across 3 LLMs

Batch mapping:
  Context Manipulation:     Claude=batch1-2, ChatGPT=batch3, Grok=batch4
  Thread Hijacking:         Claude=batch1-4, ChatGPT=batch5, Grok=batch6
  Relationship Exploitation:Claude=batch1-2, ChatGPT=batch3, Grok=batch4
  Urgency Escalation:       Claude=batch1-2, ChatGPT=batch3, Grok=batch4

Metrics:
  1. Thread structure    - num_emails, participant count, subject variation
  2. Urgency signals     - urgency keyword density
  3. Phishing markers    - suspicious domains, credential requests, pressure phrases
  4. Linguistic quality  - avg body length, tone escalation, formality drop
  5. Attack realism      - fabricated history, cross-references, entity consistency
"""

import json
import os
import re
import numpy as np
from collections import defaultdict

# ============================================================================
# BATCH SOURCE MAP
# ============================================================================

BATCH_SOURCES = {
    'context_manipulation': {
        'Claude':   ['../../data/raw/Dataset Attack/context_batch1.json', '../../data/raw/Dataset Attack/context_batch2.json'],
        'ChatGPT':  ['../../data/raw/Dataset Attack/context_batch3.json'],
        'Grok':     ['../../data/raw/Dataset Attack/context_batch4.json'],
    },
    'thread_hijacking': {
        'Claude':   ['../../data/raw/Dataset Attack/hijack_batch1.json', '../../data/raw/Dataset Attack/hijack_batch2.json',
                     '../../data/raw/Dataset Attack/hijack_batch3.json', '../../data/raw/Dataset Attack/hijack_batch4.json'],
        'ChatGPT':  ['../../data/raw/Dataset Attack/hijack_batch5.json'],
        'Grok':     ['../../data/raw/Dataset Attack/hijack_batch6.json'],
    },
    'relationship_exploitation': {
        'Claude':   ['../../data/raw/Dataset Attack/relationship_batch1.json', '../../data/raw/Dataset Attack/relationship_batch2.json'],
        'ChatGPT':  ['../../data/raw/Dataset Attack/relationship_batch3.json'],
        'Grok':     ['../../data/raw/Dataset Attack/relationship_batch4.json'],
    },
    'urgency_escalation': {
        'Claude':   ['../../data/raw/Dataset Attack/urgency_batch1.json', '../../data/raw/Dataset Attack/urgency_batch2.json'],
        'ChatGPT':  ['../../data/raw/Dataset Attack/urgency_batch3.json'],
        'Grok':     ['../../data/raw/Dataset Attack/urgency_batch4.json'],
    },
}

# ============================================================================
# PHISHING INDICATOR DICTIONARIES
# ============================================================================

URGENCY_KEYWORDS = [
    'urgent', 'asap', 'immediately', 'emergency', 'critical', 'now',
    'rush', 'hurry', 'quick', 'deadline', 'today', 'tonight', 'expires',
    'expiring', 'limited time', 'act now', 'hours', 'minutes', 'seconds',
    'final notice', 'last chance', 'time sensitive', 'do not delay'
]

PRESSURE_PHRASES = [
    'verify your', 'confirm your', 'click here', 'click the link',
    'provide your', 'enter your', 'update your', 'validate your',
    'suspended', 'terminated', 'blocked', 'locked', 'frozen',
    'unauthorized access', 'suspicious activity', 'account at risk',
    'legal action', 'arrest', 'penalty', 'fine', 'lawsuit'
]

CREDENTIAL_REQUESTS = [
    'password', 'username', 'login', 'credentials', 'ssn', 'social security',
    'bank account', 'credit card', 'card number', 'cvv', 'pin',
    'banking credentials', 'wire transfer', 'routing number'
]

SUSPICIOUS_DOMAIN_PATTERNS = [
    r'https?://[^\s]*(?:secure|verify|update|confirm|login|account|alert|fraud)[^\s]*\.[^\s]+',
    r'https?://[^\s]*-[^\s]*\.[^\s]+',   # hyphenated domains
]

FABRICATION_PHRASES = [
    'as we discussed', 'per our conversation', 'following our meeting',
    'as mentioned', 'per our agreement', 'as we agreed', 'our call',
    'our meeting', 'when we spoke', 'during our call', 'our chat',
    'as promised', 'as requested earlier'
]

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_threads(filepaths):
    threads = []
    for fp in filepaths:
        if os.path.exists(fp):
            threads.extend(json.load(open(fp, encoding='utf-8')))
    return threads


def count_keywords(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def has_suspicious_url(text):
    for pattern in SUSPICIOUS_DOMAIN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def analyze_thread(thread):
    emails = thread.get('emails', [])
    if not emails:
        return None

    bodies = [e.get('body', '') for e in emails]
    total_body = ' '.join(bodies)

    # --- Structure ---
    num_emails = len(emails)
    participants = set()
    for e in emails:
        if e.get('from'): participants.add(e['from'].lower())
        if e.get('to'):   participants.add(e['to'].lower())
    num_participants = len(participants)

    subjects = [e.get('subject', '') for e in emails]
    unique_subjects = len(set(s.lower() for s in subjects if s))

    # --- Body lengths ---
    body_lengths = [len(b.split()) for b in bodies]
    avg_body_length = np.mean(body_lengths) if body_lengths else 0

    # --- Urgency ---
    urgency_counts = [count_keywords(b, URGENCY_KEYWORDS) for b in bodies]
    urgency_density = sum(urgency_counts) / max(len(bodies), 1)
    # Escalation: does urgency increase over thread?
    urgency_escalation = urgency_counts[-1] - urgency_counts[0] if len(urgency_counts) > 1 else 0

    # --- Pressure / credential requests ---
    pressure_count = count_keywords(total_body, PRESSURE_PHRASES)
    credential_count = count_keywords(total_body, CREDENTIAL_REQUESTS)

    # --- Suspicious URLs ---
    url_count = sum(1 for b in bodies if has_suspicious_url(b))
    has_url = url_count > 0

    # --- Fabricated history ---
    fabrication_count = count_keywords(total_body, FABRICATION_PHRASES)

    # --- Sender alternation (realism of back-and-forth) ---
    senders = [e.get('from', '').lower() for e in emails]
    alternations = sum(1 for i in range(1, len(senders)) if senders[i] != senders[i-1])
    alternation_ratio = alternations / max(len(senders) - 1, 1)

    # --- Formality drop (first vs last email) ---
    formal_words = ['dear', 'sincerely', 'regards', 'respectfully', 'thank you', 'please']
    informal_words = ['hey', 'hi!', 'yo', 'gonna', 'wanna', 'btw', 'asap']
    first_formal = count_keywords(bodies[0], formal_words) if bodies else 0
    last_formal = count_keywords(bodies[-1], formal_words) if bodies else 0
    formality_drop = first_formal - last_formal

    return {
        'num_emails': num_emails,
        'num_participants': num_participants,
        'unique_subjects': unique_subjects,
        'avg_body_length_words': avg_body_length,
        'urgency_density': urgency_density,
        'urgency_escalation': urgency_escalation,
        'pressure_phrase_count': pressure_count,
        'credential_request_count': credential_count,
        'has_suspicious_url': int(has_url),
        'suspicious_url_count': url_count,
        'fabrication_phrase_count': fabrication_count,
        'sender_alternation_ratio': alternation_ratio,
        'formality_drop': formality_drop,
    }


def aggregate(metrics_list):
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: np.mean([m[k] for m in metrics_list]) for k in keys}


# ============================================================================
# SCORING: Composite "Attack Realism Score"
# ============================================================================

def realism_score(agg):
    """
    Weighted composite score (0-100) reflecting phishing attack realism.
    Higher = more realistic/effective phishing.
    """
    score = 0

    # Urgency signals (25 pts)
    score += min(25, agg['urgency_density'] * 5)

    # Pressure / credential requests (20 pts)
    score += min(10, agg['pressure_phrase_count'] * 1.0)
    score += min(10, agg['credential_request_count'] * 2.0)

    # Suspicious URLs present (15 pts)
    score += agg['has_suspicious_url'] * 15

    # Fabricated history (10 pts) - important for context manipulation
    score += min(10, agg['fabrication_phrase_count'] * 2.0)

    # Realistic thread structure (15 pts)
    # Sweet spot: 4-8 emails with back-and-forth
    email_score = min(10, max(0, agg['num_emails'] - 2) * 2)
    score += email_score
    score += min(5, agg['sender_alternation_ratio'] * 5)

    # Body length - realistic emails aren't one-liners (10 pts)
    score += min(10, agg['avg_body_length_words'] / 20)

    # Urgency escalation over thread (5 pts)
    score += min(5, max(0, agg['urgency_escalation']) * 1.5)

    return round(min(100, score), 2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("LLM PHISHING GENERATION QUALITY COMPARISON")
    print("Claude vs ChatGPT vs Grok")
    print("=" * 70)

    models = ['Claude', 'ChatGPT', 'Grok']
    attack_types = list(BATCH_SOURCES.keys())

    # Per-model overall results
    model_all_metrics = defaultdict(list)

    for attack_type in attack_types:
        print(f"\n{'='*70}")
        print(f"ATTACK TYPE: {attack_type.upper().replace('_', ' ')}")
        print(f"{'='*70}")

        model_results = {}

        for model in models:
            files = BATCH_SOURCES[attack_type][model]
            threads = load_threads(files)

            if not threads:
                print(f"  {model}: No data found")
                continue

            metrics_list = [m for t in threads if (m := analyze_thread(t)) is not None]
            agg = aggregate(metrics_list)
            score = realism_score(agg)
            model_results[model] = (agg, score, len(threads))
            model_all_metrics[model].extend(metrics_list)

        # Print comparison table for this attack type
        print(f"\n{'Metric':<32} {'Claude':>10} {'ChatGPT':>10} {'Grok':>10}")
        print("-" * 65)

        metrics_to_show = [
            ('Threads',                 'n',           False),
            ('Avg emails/thread',       'num_emails',  False),
            ('Avg participants',        'num_participants', False),
            ('Avg body length (words)', 'avg_body_length_words', False),
            ('Urgency density',         'urgency_density', False),
            ('Urgency escalation',      'urgency_escalation', False),
            ('Pressure phrases',        'pressure_phrase_count', False),
            ('Credential requests',     'credential_request_count', False),
            ('Threads with susp. URL',  'has_suspicious_url', True),
            ('Fabrication phrases',     'fabrication_phrase_count', False),
            ('Sender alternation',      'sender_alternation_ratio', False),
            ('Formality drop',          'formality_drop', False),
        ]

        for label, key, is_pct in metrics_to_show:
            row = f"  {label:<30}"
            for model in models:
                if model not in model_results:
                    row += f" {'N/A':>10}"
                    continue
                agg, score, n = model_results[model]
                if key == 'n':
                    val = n
                    row += f" {val:>10}"
                elif is_pct:
                    val = agg.get(key, 0) * 100
                    row += f" {val:>9.1f}%"
                else:
                    val = agg.get(key, 0)
                    row += f" {val:>10.2f}"
            print(row)

        print("-" * 65)
        score_row = f"  {'REALISM SCORE (0-100)':<30}"
        best_score = -1
        best_model = ''
        for model in models:
            if model in model_results:
                _, score, _ = model_results[model]
                score_row += f" {score:>10.2f}"
                if score > best_score:
                    best_score = score
                    best_model = model
            else:
                score_row += f" {'N/A':>10}"
        print(score_row)
        print(f"\n  Winner for {attack_type.replace('_',' ')}: {best_model} ({best_score:.2f}/100)")

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY (ACROSS ALL ATTACK TYPES)")
    print(f"{'='*70}")
    print(f"\n{'Metric':<32} {'Claude':>10} {'ChatGPT':>10} {'Grok':>10}")
    print("-" * 65)

    overall_scores = {}
    for model in models:
        ml = model_all_metrics[model]
        if not ml:
            continue
        agg = aggregate(ml)
        score = realism_score(agg)
        overall_scores[model] = score

        print(f"  {'Threads analyzed':<30} {len(ml):>10}")
        print(f"  {'Avg emails/thread':<30} {agg['num_emails']:>10.2f}")
        print(f"  {'Urgency density':<30} {agg['urgency_density']:>10.2f}")
        print(f"  {'Pressure phrases':<30} {agg['pressure_phrase_count']:>10.2f}")
        print(f"  {'Credential requests':<30} {agg['credential_request_count']:>10.2f}")
        print(f"  {'Threads with susp. URL':<30} {agg['has_suspicious_url']*100:>9.1f}%")
        print(f"  {'Fabrication phrases':<30} {agg['fabrication_phrase_count']:>10.2f}")
        print(f"  {'Sender alternation':<30} {agg['sender_alternation_ratio']:>10.2f}")
        print(f"  {'REALISM SCORE':<30} {score:>10.2f}")
        print()

    print("-" * 65)
    ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nFINAL RANKING:")
    for rank, (model, score) in enumerate(ranked, 1):
        print(f"  {rank}. {model:<12} {score:.2f}/100")

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("  Urgency density       : Avg urgency keywords per email (higher = more pressure)")
    print("  Pressure phrases      : Social engineering language per thread")
    print("  Credential requests   : How often the attack asks for sensitive info")
    print("  Suspicious URL        : % of threads containing phishing-style URLs")
    print("  Fabrication phrases   : References to fake prior conversations")
    print("  Sender alternation    : Realism of back-and-forth (1.0 = perfect alternation)")
    print("  Realism score         : Composite weighted score (0-100)")


if __name__ == '__main__':
    main()
