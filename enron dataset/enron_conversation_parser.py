import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
import json


class EnronConversationParser:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

    # -----------------------------------------
    def load_data(self):
        print("Loading Enron Dataset...")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"[OK] Loaded {len(self.df)} emails successfully!")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    # -----------------------------------------
    def parse_email(self, raw_message):

        email_data = {
            'message_id': None,
            'reply_to': None,
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

            elif line_lower.startswith('in-reply-to:'):
                email_data['reply_to'] = line.split(':', 1)[1].strip()

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
    def find_two_person_conversations(self, min_emails=3, sample_size=10000):
        """
        Find conversations between exactly TWO people going back and forth.
        Returns threads where Person A and Person B exchange multiple emails.
        """
        
        print(f"\nAnalyzing {sample_size} emails to find 2-person conversations...")
        
        # Group emails by subject (removing Re:, Fw: prefixes)
        subject_groups = defaultdict(list)
        
        for idx, row in self.df.head(sample_size).iterrows():
            parsed = self.parse_email(row['message'])
            
            if not parsed['subject'] or parsed['subject'] == '':
                continue
            
            # Clean subject: remove Re:, Fw:, Fwd: prefixes
            subject_clean = re.sub(r'^(re:|fw:|fwd:)\s*', '', parsed['subject'].lower(), flags=re.IGNORECASE).strip()
            
            if subject_clean:
                subject_groups[subject_clean].append({
                    'index': idx,
                    'parsed': parsed
                })
        
        print(f"Found {len(subject_groups)} unique conversation subjects")
        
        # Filter to find 2-person conversations
        two_person_threads = []
        
        for subject, emails in subject_groups.items():
            if len(emails) < min_emails:
                continue
            
            # Get unique participants
            participants = set()
            for email in emails:
                participants.add(email['parsed']['from'])
                # Also check 'to' field (might have multiple recipients, take first)
                to_addr = email['parsed']['to'].split(',')[0].strip()
                if to_addr and to_addr != 'Unknown':
                    participants.add(to_addr)
            
            # We want exactly 2 participants (A and B going back and forth)
            if len(participants) == 2:
                # Sort emails by date
                emails_sorted = sorted(emails, key=lambda x: x['parsed']['date'] if x['parsed']['date'] else pd.Timestamp.min)
                
                two_person_threads.append({
                    'subject': subject,
                    'participants': list(participants),
                    'num_emails': len(emails),
                    'emails': emails_sorted
                })
        
        print(f"[OK] Found {len(two_person_threads)} two-person conversations with {min_emails}+ emails\n")
        
        # Sort by number of emails (longest conversations first)
        two_person_threads.sort(key=lambda x: x['num_emails'], reverse=True)
        
        return two_person_threads

    # -----------------------------------------
    def display_conversation(self, thread, max_emails=10):
        """Display a conversation thread in a readable format."""
        
        print("\n" + "="*90)
        print(f"CONVERSATION: {thread['subject'][:70]}")
        print("="*90)
        print(f"Between: {thread['participants'][0]} <--> {thread['participants'][1]}")
        print(f"Total emails: {thread['num_emails']}")
        print("="*90)
        
        for i, email_info in enumerate(thread['emails'][:max_emails], 1):
            parsed = email_info['parsed']
            
            print(f"\n[EMAIL {i}/{thread['num_emails']}]")
            print("-"*90)
            print(f"From: {parsed['from']}")
            print(f"To:   {parsed['to']}")
            print(f"Date: {parsed['date']}")
            print(f"Subject: {parsed['subject']}")
            print("-"*90)
            
            # Show body (truncate if too long)
            body = parsed['body']
            if len(body) > 400:
                print(body[:400])
                print(f"\n... (truncated, {len(body) - 400} more characters)")
            else:
                print(body if body else "(Empty body)")
            
            print("-"*90)
        
        if thread['num_emails'] > max_emails:
            print(f"\n... and {thread['num_emails'] - max_emails} more emails in this conversation")
        
        print("="*90)

    # -----------------------------------------
    def build_reply_threads(self, limit=3, sample_size=15000):
        """
        Build threads using Message-ID and In-Reply-To headers.
        Shows parent-child relationships.
        """
        
        print(f"\nBuilding reply threads from {sample_size} emails...")
        
        id_map = {}
        children = defaultdict(list)
        
        for idx, row in self.df.head(sample_size).iterrows():
            mail = self.parse_email(row['message'])
            
            if mail['message_id']:
                id_map[mail['message_id']] = mail
                
                if mail['reply_to']:
                    parent = mail['reply_to']
                    children[parent].append(mail['message_id'])
        
        # Find root messages (not a reply to anything)
        roots = [mid for mid in id_map if mid not in children]
        
        print(f"Found {len(id_map)} emails with Message-IDs")
        print(f"Found {len(children)} messages with replies")
        print(f"Found {len(roots)} root messages (thread starters)\n")
        
        def count_chain_length(mid):
            """Count how many emails in this thread."""
            if mid not in id_map:
                return 0
            count = 1
            if mid in children:
                for child in children[mid]:
                    count += count_chain_length(child)
            return count
        
        # Find threads with multiple emails
        thread_roots = []
        for root in roots:
            length = count_chain_length(root)
            if length > 1:  # At least 2 emails
                thread_roots.append((root, length))
        
        # Sort by length (longest first)
        thread_roots.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(thread_roots)} threads with 2+ emails")
        print(f"Showing top {limit} longest threads:\n")
        
        def show_chain(mid, depth=0):
            """Recursively show email thread."""
            if mid not in id_map:
                return
            
            m = id_map[mid]
            indent = "   " * depth
            
            print(f"\n{indent}[EMAIL - Depth {depth}]")
            print(f"{indent}From: {m['from']}")
            print(f"{indent}To:   {m['to']}")
            print(f"{indent}Subject: {m['subject']}")
            print(f"{indent}Date: {m['date']}")
            print(f"{indent}{'-'*50}")
            
            # Show body preview
            body_preview = m['body'][:200] if m['body'] else "(Empty)"
            print(f"{indent}{body_preview}")
            if len(m['body']) > 200:
                print(f"{indent}... ({len(m['body']) - 200} more chars)")
            
            # Show replies
            if mid in children:
                print(f"{indent}[{len(children[mid])} REPL{'Y' if len(children[mid])==1 else 'IES'}]")
                for child in children[mid]:
                    show_chain(child, depth + 1)
        
        # Display threads
        count = 0
        for root, length in thread_roots[:limit]:
            print("\n" + "="*90)
            print(f"THREAD (Root Message - {length} total emails)")
            print("="*90)
            show_chain(root)
            print("="*90)
            
            count += 1
            if count >= limit:
                break
        
        return thread_roots

    # -----------------------------------------
    def export_conversation_to_json(self, thread, output_file):
        """Export a conversation to JSON format for your project."""
        
        thread_data = {
            "thread_id": f"enron_{abs(hash(thread['subject'])) % 100000}",
            "subject": thread['subject'],
            "participants": thread['participants'],
            "num_emails": thread['num_emails'],
            "label": "legitimate",
            "attack_type": None,
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
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(thread_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Conversation exported to {output_file}")
        return thread_data

    # -----------------------------------------
    def analyze_conversation_patterns(self, thread):
        """Analyze patterns in a conversation for your research."""
        
        print("\n" + "="*90)
        print("CONVERSATION ANALYSIS")
        print("="*90)
        
        emails = thread['emails']
        
        # 1. Response times
        response_times = []
        for i in range(1, len(emails)):
            if emails[i]['parsed']['date'] and emails[i-1]['parsed']['date']:
                time_diff = emails[i]['parsed']['date'] - emails[i-1]['parsed']['date']
                response_times.append(time_diff.total_seconds() / 3600)  # in hours
        
        if response_times:
            print(f"\nResponse Time Analysis:")
            print(f"  Average response time: {sum(response_times)/len(response_times):.1f} hours")
            print(f"  Fastest response: {min(response_times):.1f} hours")
            print(f"  Slowest response: {max(response_times):.1f} hours")
        
        # 2. Email length patterns
        lengths = [len(e['parsed']['body']) for e in emails]
        print(f"\nEmail Length Analysis:")
        print(f"  Average length: {sum(lengths)/len(lengths):.0f} characters")
        print(f"  Shortest email: {min(lengths)} characters")
        print(f"  Longest email: {max(lengths)} characters")
        
        # 3. Who talks more?
        person_counts = defaultdict(int)
        for e in emails:
            person_counts[e['parsed']['from']] += 1
        
        print(f"\nParticipation:")
        for person, count in person_counts.items():
            print(f"  {person}: {count} emails ({count/len(emails)*100:.1f}%)")
        
        # 4. Subject line changes
        subjects = [e['parsed']['subject'] for e in emails]
        unique_subjects = len(set(subjects))
        print(f"\nSubject Line:")
        print(f"  Subject stayed same: {'Yes' if unique_subjects == 1 else 'No'}")
        if unique_subjects > 1:
            print(f"  Changed {unique_subjects - 1} times")
        
        print("="*90)


# -----------------------------------------
def main():

    csv_path = "emails.csv"

    if not Path(csv_path).exists():
        print("[ERROR] emails.csv not found!")
        print("Please download the Enron dataset and place it in this folder.")
        return

    parser = EnronConversationParser(csv_path)

    if not parser.load_data():
        return

    conversations = None

    while True:

        print("\n" + "="*90)
        print("MENU - Enron Conversation Thread Analyzer")
        print("="*90)
        print("1. Find 2-Person Conversations (back-and-forth exchanges)")
        print("2. Show Reply Threads (using Message-ID headers)")
        print("3. Display a Conversation")
        print("4. Analyze Conversation Patterns")
        print("5. Export Conversation to JSON")
        print("6. Exit")
        print("="*90)

        ch = input("\nChoice (1-6): ").strip()

        if ch == '1':
            min_emails = input("Minimum emails in conversation (default 3): ").strip()
            min_emails = int(min_emails) if min_emails.isdigit() else 3
            
            sample_size = input("Sample size to analyze (default 10000): ").strip()
            sample_size = int(sample_size) if sample_size.isdigit() else 10000
            
            conversations = parser.find_two_person_conversations(min_emails, sample_size)
            
            if conversations:
                print("\nTop 10 longest conversations:")
                for i, conv in enumerate(conversations[:10], 1):
                    print(f"{i}. {conv['subject'][:60]}")
                    print(f"   Between: {conv['participants'][0][:30]} <--> {conv['participants'][1][:30]}")
                    print(f"   Emails: {conv['num_emails']}")
                    print()

        elif ch == '2':
            num_threads = input("How many threads to show? (default 3): ").strip()
            num_threads = int(num_threads) if num_threads.isdigit() else 3
            parser.build_reply_threads(limit=num_threads)

        elif ch == '3':
            if not conversations:
                print("\n[INFO] Please run option 1 first to find conversations!")
                continue
            
            idx = input(f"Which conversation to display? (1-{len(conversations)}): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(conversations):
                thread = conversations[int(idx) - 1]
                parser.display_conversation(thread)
            else:
                print("Invalid choice!")

        elif ch == '4':
            if not conversations:
                print("\n[INFO] Please run option 1 first to find conversations!")
                continue
            
            idx = input(f"Which conversation to analyze? (1-{len(conversations)}): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(conversations):
                thread = conversations[int(idx) - 1]
                parser.analyze_conversation_patterns(thread)
            else:
                print("Invalid choice!")

        elif ch == '5':
            if not conversations:
                print("\n[INFO] Please run option 1 first to find conversations!")
                continue
            
            idx = input(f"Which conversation to export? (1-{len(conversations)}): ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(conversations):
                thread = conversations[int(idx) - 1]
                filename = input("Output filename (default: conversation.json): ").strip()
                filename = filename if filename else "conversation.json"
                parser.export_conversation_to_json(thread, filename)
            else:
                print("Invalid choice!")

        elif ch == '6':
            print("\nGoodbye!")
            break

        else:
            print("\n[ERROR] Invalid choice! Please enter 1-6.")


if __name__ == "__main__":
    main()
