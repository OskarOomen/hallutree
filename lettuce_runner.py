import json
import logging
import os
from datetime import datetime
from load_data import load_data
from lettucedetect.models.inference import HallucinationDetector

def setup_logging():
    """Set up logging with timestamped filename"""
    # Create results/logs directory if it doesn't exist
    os.makedirs("results/logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'results/logs/lettuce_results_{timestamp}.log'
    
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
    
    logging.info(f"Lettuce logging initialized. Log file: {log_filename}")
    return log_filename

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

def run_lettuce_detect():
    # Set up logging
    log_filename = setup_logging()
    
    # Get the next run number
    run_number = get_next_run_number()
    logging.info(f"Starting Lettuce Run {run_number}")
    
    # Number of examples to test
    n_examples = 10  # Change this to test more or fewer examples
    
    # Load examples
    logging.info("Loading RAGTruth test dataset...")
    examples = load_data(n_examples)
    logging.info(f"Found {len(examples)} examples")
    
    # Store results
    results = []
    
    logging.info(f"Running LettuceDetect on {len(examples)} examples...")
    
    detector = HallucinationDetector(
        method="transformer",
        model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
    )

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
            predictions = detector.predict(
                context=[example['context']],
                question="",
                answer=example['claim'],
                output_format="spans"
            )
            prediction = len(predictions) == 0  # True if supported, False if hallucination
            result_data['success'] = True
            result_data['result'] = prediction
            logging.info(f"Lettuce prediction: {prediction} (spans: {predictions})")
            
        except Exception as e:
            result_data['error'] = str(e)
            logging.error(f"Error running LettuceDetect: {e}")
        
        results.append(result_data)
        
        # Log whether the prediction was correct
        if result_data['success']:
            predicted_supported = result_data['result']
            actual_supported = result_data['expected_supported']
            is_correct = predicted_supported == actual_supported
            status = "CORRECT" if is_correct else "INCORRECT"
            logging.info(f"Lettuce Prediction: {predicted_supported}, Expected: {actual_supported} - {status}")
        else:
            logging.info("Lettuce Prediction: FAILED (could not process)")
        
        logging.info("-" * 50)
    
    # Calculate metrics
    successful_results = [r for r in results if r['success']]
    successful_runs = len(successful_results)
    failed_runs = len(results) - successful_runs
    
    logging.info(f"\n=== LETTUCE RESULTS ===")
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
    results_file = f'results/json/lettuce_results_{run_id}.json'
    os.makedirs("results/json", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")
    logging.info(f"Log file: {log_filename}")

if __name__ == "__main__":
    run_lettuce_detect() 