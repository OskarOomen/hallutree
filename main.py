import json
from claim_tree import ClaimTree
import langsmith as ls
import os
import logging
from datetime import datetime
import io
import sys
from factcg_data_loader import load_data 
from parquet_data_loader import load_parquet_data_with_info, list_parquet_files
from tree_node import TreeNode, NodeType

os.environ["LANGSMITH_TRACING"] = "true"

def get_next_run_number():
    """Get the next run number from a file, creating it if it doesn't exist"""
    run_counter_file = "run_counter.txt"
    try:
        with open(run_counter_file, 'r') as f:
            current_run = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        current_run = 0
    next_run = current_run + 1
    with open(run_counter_file, 'w') as f:
        f.write(str(next_run))
    return next_run

def setup_logging():
    """Set up logging with timestamped filename"""
    # Create results/logs directory if it doesn't exist
    os.makedirs("results/logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'results/logs/claim_tree_results_{timestamp}.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up new logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename

def main():
    # Set up logging first
    log_filename = setup_logging()
    
    # Configuration parameters
    n_examples = 558  # Change this to test more or fewer examples
    start_index =  0   # Change this to start from a specific index (0-based)
    
    # Choose data source: 'json' or 'parquet'
    data_source = 'parquet'  # Change this to 'json' to use the original JSON loader
    
    if data_source == 'parquet':
        # Load examples from parquet file
        logging.info("Loading parquet dataset...")
        examples, file_info = load_parquet_data_with_info(n_examples, start_index=start_index)
        logging.info(f"Loaded {len(examples)} examples from {file_info['file_path']}")
        logging.info(f"File info: {file_info['total_rows']} total rows, {file_info['columns']} columns")
        logging.info(f"Range: examples {file_info['start_index']} to {file_info['end_index']-1} (inclusive)")
    else:
        # Load examples from JSON file
        logging.info("Loading JSON dataset...")
        examples = load_data(n_examples, file_path="data/training_stage1/CG2C_hotpot_qa_rbt_mnli_failed.json")
    
    logging.info(f"Found {len(examples)} examples")
    
    # Store results
    results = []

    total_extractive_claims = 0
    total_inferential_claims = 0
    
    logging.info(f"Testing on {len(examples)} examples...")
    
    # Generate unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_number = get_next_run_number()
    
    for i, example in enumerate(examples):
        trace_name = f"Run{run_number}ID{example['id']}"
        
        with ls.trace(trace_name, project_name="ragatomics2"):
            logging.info(f"\n--- Example {i+1}/{len(examples)} ---")
            logging.info(f"ID: {example['id']}")
            logging.info(f"Expected supported: {not example['is_hallucination']}")
            logging.info(f"Claim: {example['claim'][:100]}...")  # Show first 100 chars
            
            # Initialize and run ClaimTree
            result_data = {
                'id': example['id'],
                'expected_supported': not example['is_hallucination'],
                'claim': example['claim'],
                'context': example['context'],
                'success': False,
                'result': None,
                'error': None
            }
            
            try:
                tree = ClaimTree(example['claim'], example['context'])
                result = tree.construct()
                result_data['success'] = True
                result_data['result'] = result

                extractive_count = len(tree.root.find_by_type(NodeType.EXTRACTIVE))
                inferential_count = len(tree.root.find_by_type(NodeType.INFERENTIAL))
                
                total_extractive_claims += extractive_count
                total_inferential_claims += inferential_count
                
                logging.info(f"Subclaims: {extractive_count} extractive, {inferential_count} inferential")
                
                logging.info(f"ClaimTree result: {result}")
                logging.info("Tree structure:")
                
                # Capture tree structure output
                captured_output = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = captured_output
                try:
                    tree.print_tree()
                finally:
                    sys.stdout = original_stdout
                
                tree_output = captured_output.getvalue()
                logging.info(tree_output)
                
            except Exception as e:
                error_msg = f"Error processing example: {e}"
                result_data['error'] = str(e)
                logging.error(error_msg)
            
            results.append(result_data)
            
            # Log whether the prediction was correct
            if result_data['success']:
                predicted_supported = result_data['result']
                actual_supported = result_data['expected_supported']
                is_correct = predicted_supported == actual_supported
                status = "CORRECT" if is_correct else "INCORRECT"
                logging.info(f"Prediction: {predicted_supported}, Expected: {actual_supported} - {status}")
            else:
                logging.info("Prediction: FAILED (could not process)")
            
            logging.info("-" * 50)
    
    # Save results to file
    results_file = f'results/json/claim_tree_results_{run_id}.json'
    os.makedirs("results/json", exist_ok=True) # Ensure results/json directory exists
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate metrics for successful runs
    successful_results = [r for r in results if r['success']]
    successful_runs = len(successful_results)
    failed_runs = len(results) - successful_runs

    total_claims = total_extractive_claims + total_inferential_claims
    
    if successful_runs > 0:
        # Calculate precision, recall, and F1
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for result in successful_results:
            predicted_supported = result['result']  # Assuming result is boolean
            actual_supported = result['expected_supported']
            
            if predicted_supported and actual_supported:
                true_positives += 1
            elif predicted_supported and not actual_supported:
                false_positives += 1
            elif not predicted_supported and actual_supported:
                false_negatives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + (successful_runs - true_positives - false_positives - false_negatives)) / successful_runs
        
        logging.info(f"\n=== SUMMARY ===")
        logging.info(f"Total examples: {len(results)}")
        logging.info(f"Successful runs: {successful_runs}")
        logging.info(f"Failed runs: {failed_runs}")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Log file: {log_filename}")

        logging.info(f"\n=== CLAIM TYPE BREAKDOWN ===")
        logging.info(f"Total subclaims: {total_claims}")
        try:
            logging.info(f"Extractive claims: {total_extractive_claims} ({total_extractive_claims/total_claims*100:.1f}%)")
            logging.info(f"Inferential claims: {total_inferential_claims} ({total_inferential_claims/total_claims*100:.1f}%)") 
        except:
            pass

        logging.info(f"\n=== METRICS ===")
        logging.info(f"True Positives: {true_positives}")
        logging.info(f"False Positives: {false_positives}")
        logging.info(f"False Negatives: {false_negatives}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
    else:
        logging.info(f"\n=== SUMMARY ===")
        logging.info(f"Total examples: {len(results)}")
        logging.info(f"Successful runs: {successful_runs}")
        logging.info(f"Failed runs: {failed_runs}")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Log file: {log_filename}")
        logging.warning("No successful runs to calculate metrics")

if __name__ == "__main__":
    main() 