"""
Step 1: Explore Enron Fraud Dataset
This script helps you understand the structure of the downloaded dataset
"""

import pandas as pd
import os

# TODO: Update this path to where you extracted the dataset
DATASET_PATH = "enron_data_fraud_labeled.csv"


def explore_dataset():
    """Explore the basic structure of the dataset"""
    
    print("="*60)
    print("ENRON FRAUD DATASET EXPLORATION")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] File not found: {DATASET_PATH}")
        print("Please update DATASET_PATH variable with correct path")
        return
    
    # Load dataset
    print(f"\n[LOADING] Reading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Basic info
    print(f"\n[INFO] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Column names
    print(f"\n[COLUMNS] Available columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # First few rows
    print(f"\n[SAMPLE] First 3 rows:")
    print(df.head(3))
    
    # Check for fraud labels
    print(f"\n[LABELS] Checking label distribution:")
    if 'Label' in df.columns:
        print(df['Label'].value_counts())
        print(f"\nLabel meanings:")
        print(f"  0 = Legitimate (non-fraud)")
        print(f"  1 = Fraud/Phishing")
    elif 'label' in df.columns:
        print(df['label'].value_counts())
    elif 'fraud' in df.columns:
        print(df['fraud'].value_counts())
    else:
        print("No obvious label column found. Available columns:")
        print(df.columns.tolist())
    
    # Data types
    print(f"\n[TYPES] Column data types:")
    print(df.dtypes)
    
    # Missing values
    print(f"\n[MISSING] Missing values per column:")
    print(df.isnull().sum())
    
    # Email body sample
    print(f"\n[SAMPLE EMAIL] First email body:")
    body_col = None
    # Check both uppercase and lowercase
    for col in ['Body', 'body', 'message', 'Message', 'content', 'Content', 'text', 'email_body']:
        if col in df.columns:
            body_col = col
            break
    
    if body_col:
        print(f"Column: {body_col}")
        print("-" * 60)
        print(df[body_col].iloc[0][:500])  # First 500 chars
        print("-" * 60)
    else:
        print("Could not find email body column")
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = explore_dataset()
