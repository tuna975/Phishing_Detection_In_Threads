import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
import json


class EnronInternalConversationParser:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

    # -----------------------------------------
    def load_data(self):
        print("Loading Enron Dataset...")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"[OK] Loaded {len(self.df)} emails successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    # -----------------------------------------
    def parse_email(self, raw_message):
        email_data = {
            'message_id': None,
            'from': 'Unknown',
            'to': 'Unknown',
            'subject': '',
            'date': None,
            'body': ''
        }

        if pd.isna(raw_message):
            return email_data

        lines = raw_message.split('\n')
        body_start = 0

        for i, line in enumerate(lines):
            line_lower = line.lower()

            if line_lower.startswith('message-id:'):
                email_data['message_id'] = line.split(':', 1)[1].strip()
            elif line_lower.startswith('date:'):
                try:
                    email_data['date'] = pd.to_datetime(line.split(':', 1)[1].strip())
                except:
                    email_data['date'] = None
            elif line_lower.startswith('from:'):
                email_data['from'] = line.split(':', 1)[1].strip()
            elif line_lower.startswith('to:'):
                email_data['to'] = line.split(':', 1)[1].strip()
            elif line_lower.startswith('subject:'):
                email_data['subject'] = line.split(':', 1)[1].strip()
            elif line.strip() == '':
                body_start = i + 1
                break

        email_data['body'] = '\n'.join(lines[body_start:]).strip()
        return email_data

    # -----------------------------------------
    def is_enron_email(self, email_address):
        """Check if email address is from Enron domain."""
        return '@enron.com' in email_address.lower()

    # -----------------------------------------
    def find_internal_conversations(self, min_emails=3, sample_size=20000, min_exchanges=2):
        """
        Find conversations between TWO Enron employees with ACTUAL back-and-forth exchanges.
        min_exchanges: minimum number of times the conversation switches between people
        """
        
        print(f"\nAnalyzing {sample_size} emails for internal Enron conversations...")
        print("Looking for back-and-forth exchanges (Person A -> Person B -> Person A -> ...)...")
        
        # Group emails by cleaned subject
        subject_groups = defaultdict(list)
        
        for idx, row in self.df.head(sample_size).iterrows():
            parsed = self.parse_email(row['message'])
            
            if not parsed['subject']:
                continue
            
            # Only include if BOTH from and to are @enron.com
            from_addr = parsed['from'].lower()
            to_addr = parsed['to'].split(',')[0].strip().lower()  # Take first recipient
            
            if not (self.is_enron_email(from_addr) and self.is_enron_email(to_addr)):
                continue
            
            # Clean subject
            subject_clean = re.sub(r'^(re:|fw:|fwd:)\s*', '', parsed['subject'].lower(), flags=re.IGNORECASE).strip()
            
            if subject_clean:
                subject_groups[subject_clean].append({
                    'index': idx,
                    'parsed': parsed
                })
        
        print(f"Found {len(subject_groups)} conversation subjects with internal emails")
        
        # Filter for quality conversations
        quality_conversations = []
        
        for subject, emails in subject_groups.items():
            if len(emails) < min_emails:
                continue
            
            # Remove duplicate emails (same message-id or same body hash)
            seen_bodies = set()
            unique_emails = []
            
            for email in emails:
                body_hash = hash(email['parsed']['body'][:200])  # Hash first 200 chars
                if body_hash not in seen_bodies:
                    seen_bodies.add(body_hash)
                    unique_emails.append(email)
            
            if len(unique_emails) < min_emails:
                continue
            
            # Sort by date
            unique_emails.sort(key=lambda x: x['parsed']['date'] if x['parsed']['date'] else pd.Timestamp.min)
            
            # Get unique participants
            participants = set()
            for email in unique_emails:
                participants.add(email['parsed']['from'].lower())
                to_addr = email['parsed']['to'].split(',')[0].strip().lower()
                if to_addr and to_addr != 'unknown':
                    participants.add(to_addr)
            
            # Must be exactly 2 people
            if len(participants) != 2:
                continue
            
            # Check for actual back-and-forth (conversation switches)
            senders = [e['parsed']['from'].lower() for e in unique_emails]
            exchanges = sum(1 for i in range(1, len(senders)) if senders[i] != senders[i-1])
            
            if exchanges < min_exchanges:
                continue  # Not enough back-and-forth
            
            quality_conversations.append({
                'subject': subject,
                'participants': list(participants),
                'num_emails': len(unique_emails),
                'exchanges': exchanges,
                'emails': unique_emails
            })
        
        print(f"[OK] Found {len(quality_conversations)} quality conversations with {min_emails}+ emails")
        print(f"     (showing conversations with at least {min_exchanges} back-and-forth exchanges)\n")
        
        # Sort by quality: more exchanges = better
        quality_conversations.sort(key=lambda x: (x['exchanges'], x['num_emails']), reverse=True)
        
        return quality_conversations

    # -----------------------------------------
    def display_conversation(self, thread, max_emails=15):
        """Display a conversation with visual indicators for who's talking."""
        
        print("\n" + "="*100)
        print(f"CONVERSATION: {thread['subject'][:80]}")
        print("="*100)
        
        person_a, person_b = thread['participants']
        print(f"Person A: {person_a}")
        print(f"Person B: {person_b}")
        print(f"Total emails: {thread['num_emails']}")
        print(f"Back-and-forth exchanges: {thread['exchanges']}")
        print("="*100)
        
        for i, email_info in enumerate(thread['emails'][:max_emails], 1):
            parsed = email_info['parsed']
            
            # Determine who's sending
            sender = parsed['from'].lower()
            if sender == person_a:
                symbol = ">>>"
                label = "[A]"
            else:
                symbol = "<<<"
                label = "[B]"
            
            print(f"\n{symbol} EMAIL {i}/{thread['num_emails']} {label}")
            print("-"*100)
            print(f"From: {parsed['from']}")
            print(f"To:   {parsed['to']}")
            print(f"Date: {parsed['date']}")
            print("-"*100)
            
            # Show body
            body = parsed['body']
            if len(body) > 600:
                print(body[:600])
                print(f"\n... (truncated, {len(body) - 600} more characters)")
            else:
                print(body if body else "(Empty body)")
            
            print("-"*100)
        
        if thread['num_emails'] > max_emails:
            print(f"\n... and {thread['num_emails'] - max_emails} more emails in this conversation")
        
        print("="*100)

    # -----------------------------------------
    def analyze_conversation(self, thread):
        """Analyze conversation patterns."""
        
        print("\n" + "="*100)
        print("CONVERSATION ANALYSIS")
        print("="*100)
        
        emails = thread['emails']
        person_a, person_b = thread['participants']
        
        # 1. Who starts and who responds?
        print(f"\nConversation Flow:")
        print(f"  Started by: {emails[0]['parsed']['from']}")
        print(f"  Total exchanges: {thread['exchanges']}")
        
        # 2. Response times
        response_times = []
        for i in range(1, len(emails)):
            if emails[i]['parsed']['date'] and emails[i-1]['parsed']['date']:
                time_diff = emails[i]['parsed']['date'] - emails[i-1]['parsed']['date']
                response_times.append(time_diff.total_seconds() / 3600)
        
        if response_times:
            print(f"\nResponse Time Analysis:")
            print(f"  Average response: {sum(response_times)/len(response_times):.1f} hours")
            print(f"  Fastest response: {min(response_times):.1f} hours")
            print(f"  Slowest response: {max(response_times):.1f} hours")
        
        # 3. Email lengths
        lengths_a = [len(e['parsed']['body']) for e in emails if e['parsed']['from'].lower() == person_a]
        lengths_b = [len(e['parsed']['body']) for e in emails if e['parsed']['from'].lower() == person_b]
        
        print(f"\nEmail Length:")
        if lengths_a:
            print(f"  Person A average: {sum(lengths_a)/len(lengths_a):.0f} chars")
        if lengths_b:
            print(f"  Person B average: {sum(lengths_b)/len(lengths_b):.0f} chars")
        
        # 4. Participation
        count_a = len(lengths_a)
        count_b = len(lengths_b)
        
        print(f"\nParticipation:")
        print(f"  Person A sent: {count_a} emails ({count_a/len(emails)*100:.1f}%)")
        print(f"  Person B sent: {count_b} emails ({count_b/len(emails)*100:.1f}%)")
        
        balance_ratio = min(count_a, count_b) / max(count_a, count_b)
        print(f"  Balance: {balance_ratio:.2f} (1.0 = perfectly balanced)")
        
        print("="*100)

    # -----------------------------------------
    def export_to_json(self, thread, output_file):
        """Export conversation to JSON."""
        
        thread_data = {
            "thread_id": f"enron_{abs(hash(thread['subject'])) % 100000}",
            "subject": thread['subject'],
            "participants": thread['participants'],
            "num_emails": thread['num_emails'],
            "exchanges": thread['exchanges'],
            "label": "legitimate",
            "emails": []
        }
        
        for i, email_info in enumerate(thread['emails'], 1):
            parsed = email_info['parsed']
            thread_data["emails"].append({
                "email_id": i,
                "from": parsed['from'],
                "to": parsed['to'],
                "subject": parsed['subject'],
                "date": str(parsed['date']),
                "body": parsed['body']
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(thread_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Exported to {output_file}")

    # -----------------------------------------
    def batch_export(self, conversations, num_to_export=10, output_dir='conversations'):
        """Export multiple conversations at once."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\nExporting top {num_to_export} conversations to '{output_dir}/' folder...")
        
        for i, conv in enumerate(conversations[:num_to_export], 1):
            filename = f"{output_dir}/conversation_{i:02d}.json"
            self.export_to_json(conv, filename)
            print(f"  [{i}/{num_to_export}] {conv['subject'][:50]}... ({conv['num_emails']} emails, {conv['exchanges']} exchanges)")
        
        print(f"\n[OK] Exported {num_to_export} conversations successfully!")


# -----------------------------------------
def main():

    csv_path = "emails.csv"

    if not Path(csv_path).exists():
        print("[ERROR] emails.csv not found!")
        return

    parser = EnronInternalConversationParser(csv_path)

    if not parser.load_data():
        return

    conversations = None

    while True:
        print("\n" + "="*100)
        print("ENRON INTERNAL CONVERSATION ANALYZER")
        print("="*100)
        print("1. Find Internal Conversations (both people @enron.com)")
        print("2. Display a Conversation")
        print("3. Analyze Conversation Patterns")
        print("4. Export Single Conversation to JSON")
        print("5. Batch Export (top 10 conversations)")
        print("6. Exit")
        print("="*100)

        ch = input("\nChoice (1-6): ").strip()

        if ch == '1':
            min_emails = input("Minimum emails (default 3): ").strip()
            min_emails = int(min_emails) if min_emails.isdigit() else 3
            
            min_exchanges = input("Minimum back-and-forth exchanges (default 2): ").strip()
            min_exchanges = int(min_exchanges) if min_exchanges.isdigit() else 2
            
            sample_size = input("Sample size (default 20000): ").strip()
            sample_size = int(sample_size) if sample_size.isdigit() else 20000
            
            conversations = parser.find_internal_conversations(min_emails, sample_size, min_exchanges)
            
            if conversations:
                print("\nTop 15 conversations (ranked by quality):")
                print(f"{'#':<4} {'Exchanges':<12} {'Emails':<8} {'Subject':<60}")
                print("-"*100)
                for i, conv in enumerate(conversations[:15], 1):
                    print(f"{i:<4} {conv['exchanges']:<12} {conv['num_emails']:<8} {conv['subject'][:60]}")

        elif ch == '2':
            if not conversations:
                print("\n[INFO] Run option 1 first!")
                continue
            
            idx = input(f"Which conversation (1-{len(conversations)}): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(conversations):
                parser.display_conversation(conversations[int(idx) - 1])
            else:
                print("Invalid choice!")

        elif ch == '3':
            if not conversations:
                print("\n[INFO] Run option 1 first!")
                continue
            
            idx = input(f"Which conversation (1-{len(conversations)}): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(conversations):
                parser.analyze_conversation(conversations[int(idx) - 1])
            else:
                print("Invalid choice!")

        elif ch == '4':
            if not conversations:
                print("\n[INFO] Run option 1 first!")
                continue
            
            idx = input(f"Which conversation (1-{len(conversations)}): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(conversations):
                filename = input("Filename (default: conversation.json): ").strip()
                filename = filename if filename else "conversation.json"
                parser.export_to_json(conversations[int(idx) - 1], filename)
            else:
                print("Invalid choice!")

        elif ch == '5':
            if not conversations:
                print("\n[INFO] Run option 1 first!")
                continue
            
            num = input("How many to export (default 10): ").strip()
            num = int(num) if num.isdigit() else 10
            parser.batch_export(conversations, num)

        elif ch == '6':
            print("\nGoodbye!")
            break

        else:
            print("\n[ERROR] Invalid choice!")


if __name__ == "__main__":
    main()
