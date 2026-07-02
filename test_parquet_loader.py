#!/usr/bin/env python3
"""
Test script for the parquet data loader.
This script demonstrates how to use the new parquet data loader.
"""

import os
import sys
from parquet_data_loader import load_parquet_data, load_parquet_data_with_info, list_parquet_files

def test_parquet_loader():
    """Test the parquet data loader functionality."""
    
    print("=== Parquet Data Loader Test ===\n")
    
    # 1. List available parquet files
    print("1. Searching for parquet files...")
    parquet_files = list_parquet_files()
    
    if not parquet_files:
        print("   No parquet files found in data directory.")
        print("   Available files in data directory:")
        for root, dirs, files in os.walk("data"):
            for file in files:
                print(f"   - {os.path.join(root, file)}")
        return
    
    print(f"   Found {len(parquet_files)} parquet file(s):")
    for file_path in parquet_files:
        print(f"   - {file_path}")
    
    # 2. Test loading with info
    print(f"\n2. Testing data loading from {parquet_files[0]}...")
    try:
        examples, file_info = load_parquet_data_with_info(n_examples=5, file_path=parquet_files[0])
        
        print(f"   Successfully loaded {len(examples)} examples")
        print(f"   File info:")
        print(f"     - Total rows: {file_info['total_rows']}")
        print(f"     - Columns: {file_info['columns']}")
        print(f"     - File size: {file_info['file_size_mb']:.2f} MB")
        
        # 3. Display sample data
        print(f"\n3. Sample data (first 3 examples):")
        for i, example in enumerate(examples[:3]):
            print(f"   Example {i+1}:")
            print(f"     ID: {example['id']}")
            print(f"     Claim: {example['claim'][:100]}{'...' if len(example['claim']) > 100 else ''}")
            print(f"     Context (from 'doc'): {example['context'][:100]}{'...' if len(example['context']) > 100 else ''}")
            print(f"     Is hallucination (0=hallucinated, 1=factual): {example['is_hallucination']}")
            print()
        
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    print("=== Test completed successfully! ===")

if __name__ == "__main__":
    test_parquet_loader()
