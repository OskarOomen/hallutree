import json

def load_data(n_examples, file_path="RAGTruth-main/baseline/test.jsonl"):
    """Load examples from a local RAGTruth JSONL file.

    The RAGTruth dataset is not included in this repository. Download it from
    the official source and place the file at RAGTruth-main/baseline/test.jsonl,
    or pass a custom file_path.
    """
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_examples:
                break
            example = json.loads(line.strip())
            examples.append({
                'claim': example['response'],
                'context': example['reference'],
                'id': example['id'],
                'is_hallucination': len(example['labels']) > 0
            })
    return examples 