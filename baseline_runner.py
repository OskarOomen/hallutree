import json
import logging
import os
from datetime import datetime
from llm_utils import create_llm_utils
from prompts import BASELINE_HALLUCINATION_PROMPT
from parquet_data_loader import load_parquet_data_with_info, list_parquet_files

def setup_logging():
    """Set up logging with timestamped filename"""
    # Create results/logs directory if it doesn't exist
    os.makedirs("results/logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'results/logs/baseline_results_{timestamp}.log'
    
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
    
    logging.info(f"Baseline logging initialized. Log file: {log_filename}")
    return log_filename

def run_baseline():
    # Set up logging
    log_filename = setup_logging()
    
    # Number of examples to test
    n_examples = 558  # Change this to test more or fewer examples
    
    # Load examples from parquet file
    logging.info("Loading parquet dataset...")
    try:
        examples, file_info = load_parquet_data_with_info(n_examples)
        logging.info(f"Loaded {len(examples)} examples from {file_info['file_path']}")
        logging.info(f"File info: {file_info['total_rows']} total rows, {file_info['columns']} columns")
    except Exception as e:
        logging.error(f"Error loading parquet data: {e}")
        logging.info("No parquet files found or error occurred. Exiting.")
        return
    
    logging.info(f"Found {len(examples)} examples")
    
    # Initialize LLM
    llm_utils = create_llm_utils()
    
    # Store results
    results = []
    
    logging.info(f"Running baseline on {len(examples)} examples...")
    
    for i, example in enumerate(examples):
        logging.info(f"ID: {example['id']}, Expected: {not example['is_hallucination']}")
        
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
            # Run baseline prompt
            response = llm_utils.simple_completion(
                BASELINE_HALLUCINATION_PROMPT[0],
                BASELINE_HALLUCINATION_PROMPT[1].format(context=example['context'], Summary=example['claim'])
            )
            
            # Parse response
            lines = response.strip().split('\n')
            prediction = None
            for line in reversed(lines):
                line = line.strip()
                if line == "Accurate":
                    prediction = True   # Supported
                    break
                elif line == "Inaccurate":
                    prediction = False  # Not supported
                    break
            
            if prediction is not None:
                result_data['success'] = True
                result_data['result'] = prediction
                logging.info(f"Predicted: {prediction}")
            else:
                result_data['error'] = f"Could not parse response: {response}"
                logging.error(f"Could not parse response: {response}")
                
        except Exception as e:
            error_msg = f"Error processing: {e}"
            result_data['error'] = str(e)
            logging.error(error_msg)
        
        results.append(result_data)
    
    # Calculate metrics
    successful_results = [r for r in results if r['success']]
    successful_runs = len(successful_results)
    failed_runs = len(results) - successful_runs
    
    logging.info(f"\n=== BASELINE RESULTS ===")
    logging.info(f"Total examples: {len(results)}")
    logging.info(f"Successful runs: {successful_runs}")
    logging.info(f"Failed runs: {failed_runs}")
    
    if successful_runs > 0:
        # Calculate metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for result in successful_results:
            predicted_supported = result['result']
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
        
        logging.info(f"\n=== METRICS ===")
        logging.info(f"True Positives: {true_positives}")
        logging.info(f"False Positives: {false_positives}")
        logging.info(f"False Negatives: {false_negatives}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
    else:
        logging.warning("No successful runs to calculate metrics")
    
    # Save results
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/json/baseline_results_{run_id}.json'
    os.makedirs("results/json", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")
    logging.info(f"Log file: {log_filename}")

if __name__ == "__main__":
    run_baseline() 