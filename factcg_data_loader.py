import json

def load_data(n_examples, file_path="data/training_stage1/CG2C_hotpot_qa_rbt_mnli_failed.json"):
    """Load examples from a local FactCG-style JSONL file.

    The FactCG dataset is not included in this repository. Download it from
    the official source and place the file at
    data/training_stage1/CG2C_hotpot_qa_rbt_mnli_failed.json,
    or pass a custom file_path.
    """
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_examples:
                break
            example = json.loads(line.strip())
            
            # Map FactCG format to your expected format
            examples.append({
                'claim': example['text_b'][0],  # The claim is in text_b[0]
                'context': example['text_a'],   # The context is in text_a
                'id': f"factcg_{i}",           # Generate an ID since there isn't one
                'is_hallucination': example['orig_label'] == 0  # 0 = hallucination, 1 = supported
            })
    return examples