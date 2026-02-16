#!/usr/bin/env python3
"""
Verify ANN query returns correct records from specified index intervals.

Usage:
    python verify_query_intervals.py <dataset_name> <K> <num_intervals> <start1> <end1> [<start2> <end2> ...]

Example:
    python verify_query_intervals.py gist_960_euclidean_300 600 2 0 299 400 599

This will:
1. Query using embedding of idx=1 as the query vector
2. Fetch K=600 nearest neighbors using ann_distance
3. Verify all records with idx in [0,299] and [400,599] appear in results
"""

import os
import sys
import json
import requests

ASTERIX_URL = "http://localhost:19002/query/service"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def load_embedding_from_dataset(dataset_name, idx):
    """Load embedding for a specific idx from the dataset JSONL file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'datasets')
    
    # Parse dataset name to find corresponding JSONL file
    # AsterixDB dataset names use underscores: gist_960_euclidean_300
    # JSONL files use hyphens and underscores: gist-960-euclidean_train_300.jsonl
    
    # Split by last underscore to separate base name from number (if present)
    parts = dataset_name.rsplit('_', 1)
    
    possible_files = []
    
    if len(parts) == 2 and parts[1].isdigit():
        # Has a number suffix (e.g., gist_960_euclidean_300)
        base_name = parts[0].replace('_', '-')  # gist-960-euclidean
        num_records = parts[1]  # 300
        # Try: gist-960-euclidean_train_300.jsonl
        possible_files.append(os.path.join(datasets_dir, f"{base_name}_train_{num_records}.jsonl"))
        # Try: gist-960-euclidean_train.jsonl (fallback to full dataset)
        possible_files.append(os.path.join(datasets_dir, f"{base_name}_train.jsonl"))
    else:
        # No number suffix (e.g., gist_960_euclidean)
        base_name = dataset_name.replace('_', '-')
        # Try: gist-960-euclidean_train.jsonl
        possible_files.append(os.path.join(datasets_dir, f"{base_name}_train.jsonl"))
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            print(f"[info] Reading from {filepath}")
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get('idx') == idx:
                            return record.get('embedding')
                    except json.JSONDecodeError:
                        continue
            # If we read the file but didn't find the idx, continue to next file
    
    print(f"Error: Could not find embedding for idx={idx} in dataset {dataset_name}")
    print(f"Tried files: {possible_files}")
    sys.exit(1)


def execute_ann_query(dataset_name, qvec, k):
    """Execute ANN query and return results."""
    # Format embedding vector as JSON array string
    qvec_str = json.dumps(qvec)
    
    statement = f"""
    USE VectorTest;
    LET qvec = {qvec_str}
    FROM {dataset_name} row
    LET dist = ann_distance(row.embedding, qvec, "Euclidean")
    SELECT row.idx, dist
    ORDER BY dist
    LIMIT {k};
    """
    
    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'verify_query_intervals'
    }
    
    print(f"[query] Executing ANN query on {dataset_name} with K={k}")
    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error from AsterixDB: {e}")
        print(f"Response text: {resp.text}")
        sys.exit(1)
    
    try:
        result = resp.json()
        if 'results' in result:
            return result['results']
        else:
            print(f"Unexpected response format: {result}")
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response text: {resp.text}")
        sys.exit(1)


def verify_intervals(results, intervals, k):
    """
    Verify that:
    1. We got exactly K results
    2. All records from specified intervals appear in results
    
    Returns: (success: bool, message: str)
    """
    # Check 1: Result count
    if len(results) != k:
        return False, f"Expected {k} results, got {len(results)}"
    
    # Extract idx values from results
    result_indices = set()
    for record in results:
        result_indices.add(record['idx'])
    
    # Build expected indices from intervals
    expected_indices = set()
    for start, end in intervals:
        for idx in range(start, end + 1):
            expected_indices.add(idx)
    
    # Check 2: All expected indices present
    missing_indices = expected_indices - result_indices
    if missing_indices:
        missing_list = sorted(list(missing_indices))
        return False, f"Missing {len(missing_indices)} expected indices: {missing_list[:10]}{'...' if len(missing_list) > 10 else ''}"
    
    # Check for unexpected indices (indices outside intervals)
    unexpected_indices = result_indices - expected_indices
    if unexpected_indices:
        unexpected_list = sorted(list(unexpected_indices))
        return False, f"Found {len(unexpected_indices)} unexpected indices (outside intervals): {unexpected_list[:10]}{'...' if len(unexpected_list) > 10 else ''}"
    
    return True, f"✓ All {len(expected_indices)} expected indices found in results"


def main():
    if len(sys.argv) < 5:
        print("Usage: python verify_query_intervals.py <dataset_name> <K> <num_intervals> <start1> <end1> [<start2> <end2> ...]")
        print("Example: python verify_query_intervals.py gist_960_euclidean_300 600 2 0 299 400 599")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    k = int(sys.argv[2])
    num_intervals = int(sys.argv[3])
    
    # Parse intervals
    if len(sys.argv) != 4 + (num_intervals * 2):
        print(f"Error: Expected {num_intervals * 2} interval values (start/end pairs), got {len(sys.argv) - 4}")
        sys.exit(1)
    
    intervals = []
    for i in range(num_intervals):
        start_idx = 4 + (i * 2)
        end_idx = 4 + (i * 2) + 1
        start = int(sys.argv[start_idx])
        end = int(sys.argv[end_idx])
        intervals.append((start, end))
    
    print("=" * 60)
    print("ANN Query Interval Verification")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"K: {k}")
    print(f"Intervals: {intervals}")
    print()
    
    # Step 1: Load query vector (using idx=1)
    query_idx = 1
    print(f"[step] Loading embedding for idx={query_idx} as query vector")
    qvec = load_embedding_from_dataset(dataset_name, query_idx)
    if qvec is None:
        print(f"Error: Could not load embedding for idx={query_idx}")
        sys.exit(1)
    print(f"[info] Loaded embedding with dimension {len(qvec)}")
    print()
    
    # Step 2: Execute query
    results = execute_ann_query(dataset_name, qvec, k)
    print(f"[info] Received {len(results)} results")
    print()
    
    # Step 3: Verify results
    print("[step] Verifying results...")
    success, message = verify_intervals(results, intervals, k)
    
    print()
    print("=" * 60)
    if success:
        print("✓ VERIFICATION PASSED")
        print(message)
    else:
        print("✗ VERIFICATION FAILED")
        print(message)
    print("=" * 60)
    
    # Print sample of results
    print()
    print("Sample of results (first 10):")
    for i, record in enumerate(results[:10]):
        print(f"  {i+1}. idx={record['idx']}, dist={record['dist']}")
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
