import re
import sys
import os
from typing import List, Dict, Tuple

def parse_prediction_line(line: str) -> Tuple[bool, bool, bool]:
    """
    Parse a prediction line like "Prediction: True, Expected: True - CORRECT"
    
    Args:
        line: The prediction line to parse
        
    Returns:
        Tuple of (predicted_supported, actual_supported, is_correct)
    """
    # Pattern to match: "Prediction: True, Expected: True - CORRECT"
    pattern = r'Prediction:\s*(True|False),\s*Expected:\s*(True|False)\s*-\s*(CORRECT|INCORRECT)'
    match = re.search(pattern, line)
    
    if not match:
        return None, None, None
    
    predicted_str = match.group(1)
    expected_str = match.group(2)
    status = match.group(3)
    
    predicted_supported = predicted_str.lower() == 'true'
    actual_supported = expected_str.lower() == 'true'
    is_correct = status == 'CORRECT'
    
    return predicted_supported, actual_supported, is_correct

def calculate_metrics(results: List[Tuple[bool, bool, bool]]) -> Dict[str, float]:
    """
    Calculate metrics from the results.
    
    Args:
        results: List of (predicted_supported, actual_supported, is_correct) tuples
        
    Returns:
        Dictionary containing the metrics
    """
    if not results:
        return {
            'total_examples': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    total_examples = len(results)
    successful_runs = len(results)
    failed_runs = 0  # We only count successful predictions in the log
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for predicted_supported, actual_supported, is_correct in results:
        if predicted_supported and actual_supported:
            true_positives += 1
        elif predicted_supported and not actual_supported:
            false_positives += 1
        elif not predicted_supported and actual_supported:
            false_negatives += 1
        # else: true_negatives (both False) - not counted in these metrics
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + (successful_runs - true_positives - false_positives - false_negatives)) / successful_runs
    
    return {
        'total_examples': total_examples,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def calculate_metrics_range(results: List[Tuple[bool, bool, bool]], start_idx: int, end_idx: int) -> Dict[str, float]:
    """
    Calculate metrics for a subset of results from start_idx to end_idx (inclusive).
    
    Args:
        results: List of prediction results
        start_idx: Starting index (0-based)
        end_idx: Ending index (inclusive, 0-based)
    
    Returns:
        Dictionary of calculated metrics for that range
    """
    if not results:
        return calculate_metrics([])

    # Clamp indices to valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(results) - 1, end_idx)

    if start_idx > end_idx:
        return calculate_metrics([])

    subset = results[start_idx:end_idx + 1]
    return calculate_metrics(subset)

def parse_log_file(log_file_path: str) -> List[Tuple[bool, bool, bool]]:
    """
    Parse a log file and extract prediction results.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        List of (predicted_supported, actual_supported, is_correct) tuples
    """
    results = []
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found: {log_file_path}")
        return results
    
    with open(log_file_path, 'r', encoding='ISO-8859-1') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Look for prediction lines
            if 'Prediction:' in line and 'Expected:' in line and ('CORRECT' in line or 'INCORRECT' in line):
                result = parse_prediction_line(line)
                if result[0] is not None:  # Valid prediction line
                    results.append(result)
                    print(f"Line {line_num}: {line}")
    
    return results

def main():
    """Main function to parse log file and display metrics."""
    """Main function to parse log file and display metrics."""
    if len(sys.argv) not in (2, 4):
        print("Usage: python parse_log_metrics.py <log_file_path> [<start_index> <end_index>]")
        print("Example: python parse_log_metrics.py results/log.log")
        print("Example: python parse_log_metrics.py results/log.log 10 50")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    
    print(f"Parsing log file: {log_file_path}")
    print("=" * 50)
    
    # Parse the log file
    results = parse_log_file(log_file_path)
    
    if not results:
        print("No prediction results found in the log file.")
        sys.exit(1)
    
    print(f"\nFound {len(results)} prediction results.")
    print("=" * 50)
    
    if len(sys.argv) == 4:
        start_idx = int(sys.argv[2])
        end_idx = int(sys.argv[3])
        metrics = calculate_metrics_range(results, start_idx, end_idx)
        print(f"\n=== SUMMARY (Entries {start_idx} to {end_idx}) ===")
    else:
        metrics = calculate_metrics(results)
        print(f"\n=== SUMMARY (All entries) ===")
    
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Successful runs: {metrics['successful_runs']}")
    print(f"Failed runs: {metrics['failed_runs']}")
    
    print(f"\n=== METRICS ===")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    correct_predictions = sum(1 for _, _, is_correct in results if is_correct)
    incorrect_predictions = len(results) - correct_predictions
    
    print(f"\n=== PREDICTION BREAKDOWN ===")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Success rate: {correct_predictions/len(results)*100:.1f}%")

if __name__ == "__main__":
    main()