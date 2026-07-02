import pandas as pd
import os
from typing import List, Dict, Any

def load_parquet_data(n_examples: int, file_path: str = "data/test-00000-of-00001.parquet", start_index: int = 0) -> List[Dict[str, Any]]:
    """
    Load n examples from a parquet file starting from a specific index.
    
    Parquet datasets are not included in this repository. Download the relevant
    benchmark dataset separately and place it at the default path, or pass a
    custom file_path.
    
    Args:
        n_examples: Number of examples to load
        file_path: Path to the parquet file (relative to project root)
        start_index: Index to start loading from (0-based)
    
    Returns:
        List of dictionaries containing the examples with keys:
        - 'claim': The claim text
        - 'context': The context text (mapped from 'doc' column)
        - 'id': Example ID
        - 'is_hallucination': Boolean indicating if it's a hallucination
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Check if start_index is valid
    if start_index >= len(df):
        raise ValueError(f"Start index {start_index} is greater than or equal to the total number of rows {len(df)}")
    
    if start_index < 0:
        raise ValueError(f"Start index {start_index} cannot be negative")
    
    # Take n examples starting from start_index
    end_index = min(start_index + n_examples, len(df))
    df_subset = df.iloc[start_index:end_index]
    
    examples = []
    for i, row in df_subset.iterrows():
        example = {}
        
        # Map claim - use 'claim' column
        if 'claim' in df.columns:
            example['claim'] = str(row['claim'])
        else:
            raise ValueError(f"Could not find 'claim' column in parquet file. Available columns: {list(df.columns)}")
        
        # Map context - use 'doc' column
        if 'doc' in df.columns:
            example['context'] = str(row['doc'])
        else:
            raise ValueError(f"Could not find 'doc' column in parquet file. Available columns: {list(df.columns)}")
        
        # Map ID
        if 'id' in df.columns:
            example['id'] = str(row['id'])
        else:
            example['id'] = f"parquet_{i}"
        
        # Map hallucination label - use 'label' column where 1 is factual, 0 is hallucinated
        if 'label' in df.columns:
            label = row['label']
            # 1 = factual (not hallucination), 0 = hallucinated
            example['is_hallucination'] = label == 0
        else:
            # Default to False if no label column found
            example['is_hallucination'] = False
        
        examples.append(example)
    
    return examples

def list_parquet_files(data_dir: str = "data") -> List[str]:
    """
    List all parquet files in the data directory and subdirectories.
    
    Args:
        data_dir: Directory to search for parquet files
    
    Returns:
        List of paths to parquet files
    """
    parquet_files = []
    
    if not os.path.exists(data_dir):
        return parquet_files
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    return parquet_files

def load_parquet_data_with_info(n_examples: int, file_path: str = None, start_index: int = 0) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load parquet data with additional information about the file.
    
    Args:
        n_examples: Number of examples to load
        file_path: Path to the parquet file (if None, will use the first parquet file found)
        start_index: Index to start loading from (0-based)
    
    Returns:
        Tuple of (examples, file_info)
    """
    if file_path is None:
        # Find the first parquet file
        parquet_files = list_parquet_files()
        if not parquet_files:
            raise FileNotFoundError("No parquet files found in data directory")
        file_path = parquet_files[0]
    
    # Load the data
    examples = load_parquet_data(n_examples, file_path, start_index)
    
    # Get file info
    df = pd.read_parquet(file_path)
    file_info = {
        'file_path': file_path,
        'total_rows': len(df),
        'columns': list(df.columns),
        'loaded_examples': len(examples),
        'start_index': start_index,
        'end_index': start_index + len(examples),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
    }
    
    return examples, file_info
