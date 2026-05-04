"""
Thread Viewer for Cleaned V2 Threads
View deduplicated threads with proper timestamps
"""

import json
import os
import sys

# Configuration
TOP_150_FILE = "cleaned_threads_v2/top_150_for_dataset.json"
TOP_200_FILE = "cleaned_threads_v2/top_200_threads.json"
ALL_THREADS_FILE = "cleaned_threads_v2/deduplicated_threads.json"

def display_thread(thread, thread_id):
    """Display a single thread with all emails and timestamps"""
    
    print("\n" + "="*80)
    print(f"THREAD: {thread_id}")
    print("="*80)
    
    # Thread metadata
    print(f"Subject: {thread['subject']}")
    print(f"Emails in thread: {thread['num_emails']}")
    print(f"Quality Score: {thread['quality_score']}/100")
    print(f"Participants: {', '.join([p.split('@')[0] for p in thread.get('participants', [])])}")
    
    if thread.get('duplicates_removed', 0) > 0:
        print(f"Duplicates removed: {thread['duplicates_removed']}")
    
    date_start = thread.get('date_start', 'N/A')[:19]
    date_end = thread.get('date_end', 'N/A')[:19]
    print(f"Date range: {date_start} to {date_end}")
    
    print("\n" + "-"*80)
    print("EMAILS (WITH TIMESTAMPS):")
    print("-"*80)
    
    # Display each email
    emails = thread.get('emails', [])
    for i, email in enumerate(emails, 1):
        # Extract info
        sender = email.get('from', 'unknown')
        recipient = email.get('to', 'unknown')
        subject = email.get('subject', 'No subject')
        
        # Get timestamp - try parsed version first, then original
        timestamp = email.get('timestamp_parsed', email.get('date', 'unknown'))[:19]
        
        body = email.get('body', '')
        
        # Shorten email addresses
        sender_short = sender.split('@')[0] if '@' in sender else sender
        recipient_short = recipient.split('@')[0] if '@' in recipient else recipient
        
        print(f"\n[Email {i}/{len(emails)}]")
        print(f"Timestamp: {timestamp}")
        print(f"From: {sender_short} → To: {recipient_short}")
        print(f"Subject: {subject}")
        print(f"\nBody:")
        
        # Show body (first 400 chars)
        if body:
            body_preview = body[:400].strip()
            # Clean up the preview
            body_preview = ' '.join(body_preview.split())
            print(body_preview)
            if len(body) > 400:
                print(f"\n... [truncated, {len(body) - 400} more characters]")
        else:
            print("[No body content]")
        
        print("-"*80)
    
    print("\n")

def search_threads(threads, keyword):
    """Search for threads containing a keyword"""
    
    keyword = keyword.lower()
    matching_threads = {}
    
    for thread_id, thread in threads.items():
        # Search in subject
        if keyword in thread.get('subject', '').lower():
            matching_threads[thread_id] = thread
            continue
        
        # Search in participant names
        participants = thread.get('participants', [])
        if any(keyword in p.lower() for p in participants):
            matching_threads[thread_id] = thread
            continue
        
        # Search in emails
        for email in thread.get('emails', []):
            body = email.get('body', '').lower()
            subject = email.get('subject', '').lower()
            
            if keyword in body or keyword in subject:
                matching_threads[thread_id] = thread
                break
    
    return matching_threads

def interactive_viewer(threads_file):
    """
    Interactive viewer for browsing threads
    """
    
    # Load threads
    print("\nLoading threads...")
    with open(threads_file, 'r', encoding='utf-8') as f:
        threads = json.load(f)
    
    thread_list = list(threads.items())
    
    print(f"\n{'='*80}")
    print(f"THREAD VIEWER - {len(thread_list):,} threads loaded")
    print(f"{'='*80}")
    
    current_idx = 0
    current_view = thread_list  # Current list being viewed
    
    while True:
        # Show current thread
        if not current_view:
            print("\nNo threads to display.")
            current_view = thread_list
            current_idx = 0
            continue
        
        thread_id, thread = current_view[current_idx]
        display_thread(thread, thread_id)
        
        # Show navigation
        print(f"Viewing thread {current_idx + 1} of {len(current_view):,}")
        print("\nCommands:")
        print("  [n]ext / [Enter] - Next thread")
        print("  [p]revious       - Previous thread")
        print("  [j]ump NUM       - Jump to thread number NUM")
        print("  [s]earch         - Search threads by keyword")
        print("  [t]op N          - Show top N threads")
        print("  [r]eset          - Reset to full list")
        print("  [q]uit           - Exit viewer")
        
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd == 'n' or cmd == '':
            current_idx = (current_idx + 1) % len(current_view)
        
        elif cmd == 'p':
            current_idx = (current_idx - 1) % len(current_view)
        
        elif cmd.startswith('j'):
            try:
                parts = cmd.split()
                if len(parts) == 2:
                    jump_num = int(parts[1])
                else:
                    jump_num = int(input("Jump to thread number: "))
                
                if 1 <= jump_num <= len(current_view):
                    current_idx = jump_num - 1
                else:
                    print(f"Invalid number. Must be 1-{len(current_view)}")
                    input("Press Enter to continue...")
            except ValueError:
                print("Invalid number")
                input("Press Enter to continue...")
        
        elif cmd == 's':
            keyword = input("Enter search keyword: ").strip()
            if keyword:
                matching = search_threads(dict(current_view), keyword)
                if matching:
                    current_view = list(matching.items())
                    current_idx = 0
                    print(f"\nFound {len(matching)} threads matching '{keyword}'")
                    input("Press Enter to continue...")
                else:
                    print(f"\nNo threads found matching '{keyword}'")
                    input("Press Enter to continue...")
        
        elif cmd.startswith('t'):
            try:
                parts = cmd.split()
                if len(parts) == 2:
                    top_n = int(parts[1])
                else:
                    top_n = int(input("Show top N threads (N=?): "))
                
                current_view = thread_list[:top_n]
                current_idx = 0
                print(f"\nShowing top {top_n} threads")
                input("Press Enter to continue...")
            except ValueError:
                print("Invalid number")
                input("Press Enter to continue...")
        
        elif cmd == 'r':
            current_view = thread_list
            current_idx = 0
            print("\nReset to full thread list")
            input("Press Enter to continue...")
        
        elif cmd == 'q':
            print("\nExiting viewer...")
            break
        
        else:
            print(f"Unknown command: {cmd}")
            input("Press Enter to continue...")

def quick_view_threads(threads_file, n=10):
    """
    Quickly view top N threads
    """
    
    print(f"\nLoading top {n} threads...")
    with open(threads_file, 'r', encoding='utf-8') as f:
        threads = json.load(f)
    
    thread_list = list(threads.items())[:n]
    
    for i, (thread_id, thread) in enumerate(thread_list, 1):
        display_thread(thread, thread_id)
        
        if i < len(thread_list):
            response = input(f"\nPress Enter for next thread, or 'q' to quit: ")
            if response.lower() == 'q':
                break

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("CLEANED THREADS VIEWER V2")
    print("="*80)
    print("\nWhich dataset would you like to view?")
    print("  1. Top 150 threads (for dataset) - RECOMMENDED")
    print("  2. Top 200 threads")
    print("  3. All 2,622 cleaned threads")
    
    dataset_choice = input("\nEnter choice (1/2/3): ").strip()
    
    if dataset_choice == '1':
        threads_file = TOP_150_FILE
        print("\nViewing: Top 150 threads for dataset")
    elif dataset_choice == '2':
        threads_file = TOP_200_FILE
        print("\nViewing: Top 200 threads")
    elif dataset_choice == '3':
        threads_file = ALL_THREADS_FILE
        print("\nViewing: All cleaned threads")
    else:
        print("Invalid choice. Using Top 150...")
        threads_file = TOP_150_FILE
    
    if not os.path.exists(threads_file):
        print(f"\n[ERROR] File not found: {threads_file}")
        print("Please run step4_deduplicate_v2.py first")
        sys.exit(1)
    
    print("\nViewing mode:")
    print("  1. Interactive viewer (browse with commands)")
    print("  2. Quick view (show top 5 threads)")
    print("  3. Quick view (show top 10 threads)")
    
    mode_choice = input("\nEnter choice (1/2/3): ").strip()
    
    if mode_choice == '1':
        interactive_viewer(threads_file)
    elif mode_choice == '2':
        quick_view_threads(threads_file, 5)
    elif mode_choice == '3':
        quick_view_threads(threads_file, 10)
    else:
        print("Invalid choice. Running interactive viewer...")
        interactive_viewer(threads_file)