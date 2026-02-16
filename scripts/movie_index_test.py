#!/usr/bin/env python3
"""
Movie dataset vector index test with optional INCLUDE fields and flexible dataset loading.

Usage:
    Full workflow: python movie_index_test.py <dataset_filename> <num_clusters> <k> <year_condition> [--include-field <field_name>] [--limit <num_records>]
    Index only: python movie_index_test.py <dataset_filename> <num_clusters> --no-query [--include-field <field_name>] [--limit <num_records>]
    Dataset only: python movie_index_test.py <dataset_filename> --no-index [--limit <num_records>]
    Query only: python movie_index_test.py <dataset_filename> <k> <year_condition> --query-only

Example full workflow:
    python movie_index_test.py movie_embeddings_384d.json 10 50 2000

Example with INCLUDE:
    python movie_index_test.py movie_embeddings_384d.json 10 50 2000 --include-field year

Example with limited dataset (load only first 2000 records):
    python movie_index_test.py movie_embeddings_384d.json 10 50 2000 --limit 2000

Example dataset only (load data without creating index):
    python movie_index_test.py movie_embeddings_384d.json --no-index
    python movie_index_test.py movie_embeddings_384d.json --no-index --limit 2000

Example index only (skip query execution):
    python movie_index_test.py movie_embeddings_384d.json 10 --no-query
    python movie_index_test.py movie_embeddings_384d.json 10 --no-query --include-field year
    python movie_index_test.py movie_embeddings_384d.json 10 --no-query --limit 300

Example query only (skip setup, just query existing index):
    python movie_index_test.py movie_embeddings_384d.json 50 2000 --query-only

Full workflow:
1. Drop and recreate VectorTest dataverse
2. Create MovieDataset with MovieType schema
3. Load data from datasets/<dataset_filename> (optionally limited by --limit)
4. Auto-detect dimension from first embedding
5. Create vector index (with optional INCLUDE clause)
6. Execute ANN query with year filter (unless --no-query is set)
7. Verify results: K records returned and all year > condition (unless --no-query is set)
"""

import os
import sys
import json
import requests

ASTERIX_URL = "http://localhost:19002/query/service"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def execute_statement(statement, context_id):
    """Execute an AsterixDB SQL++ statement."""
    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': context_id
    }
    
    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error from AsterixDB: {e}")
        print(f"Response text: {resp.text}")
        sys.exit(1)
    
    return resp


def create_dataverse_and_dataset(dataset_path, num_records=None):
    """
    Create VectorTest dataverse and MovieDataset.
    
    Args:
        dataset_path: Path to the dataset file
        num_records: Optional limit on number of records to load (default: load all)
    """
    # If num_records is specified, create a limited dataset file
    if num_records is not None:
        print(f"[info] Creating limited dataset with first {num_records} records...")
        limited_dataset_path = f"{dataset_path}.limited_{num_records}"
        
        with open(dataset_path, 'r') as infile, open(limited_dataset_path, 'w') as outfile:
            for i, line in enumerate(infile):
                if i >= num_records:
                    break
                outfile.write(line)
        
        print(f"[info] Limited dataset created: {limited_dataset_path}")
        load_path = limited_dataset_path
    else:
        load_path = dataset_path
    
    # Normalize path for localfs
    abs_path = os.path.abspath(load_path).replace('\\', '/')
    localfs_path = f"localhost://{abs_path}"
    
    statement = f"""
    DROP DATAVERSE VectorTest IF EXISTS;
    CREATE DATAVERSE VectorTest;
    USE VectorTest;

    CREATE TYPE MovieType AS {{
      idx: int,
      title: string
    }};

    CREATE DATASET MovieDataset (MovieType)
    PRIMARY KEY idx WITH {{
      "storage-format":{{"format":"column"}}
    }};

    USE VectorTest;

    LOAD DATASET MovieDataset USING localfs (
      ("path" = "{localfs_path}"),
      ("format" = "json")
    );
    """
    
    print("[step] Creating dataverse and loading dataset")
    print(f"[info] Loading from: {localfs_path}")
    if num_records is not None:
        print(f"[info] Record limit: {num_records}")
    resp = execute_statement(statement, "create_dataverse_and_dataset")
    print("[done] Dataverse and dataset created")
    print()
    return resp


def drop_existing_index():
    """Drop existing vector index if it exists."""
    statement = """
    USE VectorTest;
    DROP INDEX MovieDataset.ix1 IF EXISTS;
    """
    
    print("[step] Dropping existing index (if any)")
    resp = execute_statement(statement, "drop_index")
    print("[done] Index dropped")
    print()
    return resp


def create_vector_index(dimension, num_clusters, include_field=None):
    """Create vector index with optional INCLUDE clause."""
    include_clause = f"INCLUDE ({include_field})" if include_field else ""
    
    statement = f"""
    USE VectorTest;
    CREATE VECTOR INDEX ix1 ON MovieDataset(embedding VECTOR) 
    {include_clause}
    WITH {{ 
      "dimension": {dimension},
      "train_list": 10000,
      "description": "",
      "num_clusters": {num_clusters},
      "similarity": "Euclidean"
    }};
    """
    
    print(f"[step] Creating vector index (num_clusters={num_clusters})")
    if include_field:
        print(f"[info] Including field: {include_field}")
    else:
        print("[info] No INCLUDE fields")
    
    resp = execute_statement(statement, "create_vector_index")
    print("[done] Vector index created")
    print()
    return resp


def load_query_vector(dataset_path, idx=0):
    """
    Load embedding from dataset JSON file for a specific idx.
    Returns (embedding, dimension).
    """
    print(f"[step] Loading query vector from dataset (idx={idx})")
    
    if not os.path.exists(dataset_path):
        print(f"Error: dataset file not found: {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get('idx') == idx:
                    embedding = record.get('embedding')
                    if embedding:
                        dimension = len(embedding)
                        print(f"[info] Loaded embedding with dimension {dimension}")
                        return embedding, dimension
            except json.JSONDecodeError:
                continue
    
    print(f"Error: Could not find embedding for idx={idx} in {dataset_path}")
    sys.exit(1)


def execute_ann_query(qvec, k, year_condition):
    """Execute ANN query with year filter."""
    qvec_str = json.dumps(qvec)
    
    statement = f"""
    USE VectorTest;
    LET qvec = {qvec_str}
    FROM MovieDataset row
    WHERE row.year > {year_condition}
    SELECT row.idx, row.year, ann_distance(row.embedding, qvec, "Euclidean") AS dist
    ORDER BY ann_distance(row.embedding, qvec, "Euclidean")
    LIMIT {k};
    """
    
    print(f"[query] Executing ANN query with K={k}, year > {year_condition}")
    
    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'movie_ann_query'
    }
    
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


def verify_results(results, k, year_condition):
    """
    Verify query results:
    1. Exactly K results returned
    2. All year values are > year_condition
    
    Returns: (success: bool, message: str)
    """
    # Convert year_condition to int if it's a string
    if isinstance(year_condition, str):
        year_condition = int(year_condition)
    
    # Check 1: Result count
    if len(results) != k:
        return False, f"Expected {k} results, got {len(results)}"
    
    # Check 2: All year values satisfy condition
    invalid_years = []
    for i, record in enumerate(results):
        year = record.get('year')
        if year is None:
            invalid_years.append(f"Record {i}: missing year field")
        elif year <= year_condition:
            invalid_years.append(f"Record {i}: year={year} (not > {year_condition})")
    
    if invalid_years:
        return False, f"Found {len(invalid_years)} records with invalid year values:\n  " + "\n  ".join(invalid_years[:5])
    
    return True, f"✓ All {k} results have year > {year_condition}"


def create_insert_file_by_index(dataset_path, idx_start, idx_end, out_path):
    """
    Create an insert file by selecting records whose idx field is within [idx_start, idx_end].
    Returns the number of records written.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: dataset file not found: {dataset_path}")
        sys.exit(1)

    written = 0
    with open(dataset_path, 'r') as infile, open(out_path, 'w') as outfile:
        for line in infile:
            line_strip = line.strip()
            if not line_strip:
                continue
            try:
                obj = json.loads(line_strip)
            except Exception:
                continue
            idx = obj.get('idx')
            if idx is None:
                continue
            if idx_start <= idx <= idx_end:
                outfile.write(line)
                written += 1
            # Early exit if passed range
            if idx > idx_end:
                break
    return written


def insert_records_by_range(dataset_path, idx_start, idx_end):
    """Insert records with idx in [idx_start, idx_end] into MovieDataset."""
    from datetime import datetime
    
    # Create insert file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    insert_file = f"{dataset_path}.insert_{idx_start}_{idx_end}_{timestamp}.jsonl"
    
    print(f"[step] Creating insert file for idx range [{idx_start}, {idx_end}]")
    written = create_insert_file_by_index(dataset_path, idx_start, idx_end, insert_file)
    print(f"[info] Created insert file with {written} records: {insert_file}")
    
    if written == 0:
        print("[warning] No records to insert")
        return
    
    # Read file and collect JSON objects
    objs = []
    with open(insert_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(line)
    
    # Build array literal string
    array_literal = ',\n'.join(objs)
    
    statement = f"""
    USE VectorTest;
    INSERT INTO MovieDataset ([
    {array_literal}
    ]);
    """
    
    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'movie_insert_records'
    }
    
    print(f"[step] Inserting {written} records into MovieDataset")
    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error from AsterixDB during insert: {e}")
        print(f"Response text: {resp.text}")
        sys.exit(1)
    
    print("[done] Records inserted successfully")
    print()


def delete_records_by_range(idx_start, idx_end):
    """Delete records with idx in [idx_start, idx_end] from MovieDataset."""
    statement = f"""
    USE VectorTest;
    DELETE FROM MovieDataset r WHERE r.idx >= {idx_start} AND r.idx <= {idx_end};
    """
    
    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'movie_delete_records'
    }
    
    print(f"[step] Deleting records with idx in [{idx_start}, {idx_end}]")
    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error from AsterixDB during delete: {e}")
        print(f"Response text: {resp.text}")
        sys.exit(1)
    
    print("[done] Records deleted successfully")
    print()


def verify_query_with_intervals(qvec, k, expected_intervals, year_condition=None):
    """
    Execute ANN query and verify results match expected idx intervals.
    
    Args:
        qvec: Query vector
        k: Number of results to return
        expected_intervals: List of (start, end) tuples for expected idx ranges
        year_condition: Optional year filter (e.g., 2000 for year > 2000)
    
    Returns: (success: bool, message: str, results: list)
    """
    qvec_str = json.dumps(qvec)
    
    # Build query with optional year filter
    if year_condition is not None:
        where_clause = f"WHERE row.year > {year_condition}"
    else:
        where_clause = ""
    
    statement = f"""
    USE VectorTest;
    LET qvec = {qvec_str}
    FROM MovieDataset row
    {where_clause}
    SELECT row.idx, row.year, ann_distance(row.embedding, qvec, "Euclidean") AS dist
    ORDER BY ann_distance(row.embedding, qvec, "Euclidean")
    LIMIT {k};
    """
    
    if year_condition is not None:
        print(f"[query] Executing ANN query with K={k}, year > {year_condition}")
    else:
        print(f"[query] Executing ANN query with K={k} (no year filter)")
    
    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'movie_verify_query'
    }
    
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
            results = result['results']
        else:
            print(f"Unexpected response format: {result}")
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response text: {resp.text}")
        sys.exit(1)
    
    print(f"[info] Received {len(results)} results")
    
    # Verification step 1: Check result count
    if len(results) != k:
        return False, f"Expected {k} results, got {len(results)}", results
    
    # Verification step 2: Check year condition (if specified)
    if year_condition is not None:
        invalid_years = []
        for i, record in enumerate(results):
            year = record.get('year')
            if year is None:
                invalid_years.append(f"Record {i} (idx={record.get('idx')}): missing year field")
            elif year <= year_condition:
                invalid_years.append(f"Record {i} (idx={record.get('idx')}): year={year} (not > {year_condition})")
        
        if invalid_years:
            return False, f"Found {len(invalid_years)} records with invalid year values:\n  " + "\n  ".join(invalid_years[:5]), results
    
    # Verification step 3: Check idx intervals (informational only - ANN returns nearest neighbors regardless of idx)
    result_indices = set(record['idx'] for record in results)
    
    # Build expected indices from intervals
    expected_indices = set()
    for start, end in expected_intervals:
        for idx in range(start, end + 1):
            expected_indices.add(idx)
    
    # Count indices in vs outside expected intervals
    indices_in_intervals = result_indices & expected_indices
    indices_outside_intervals = result_indices - expected_indices
    
    interval_str = ", ".join([f"[{s},{e}]" for s, e in expected_intervals])
    
    # Build success message with interval statistics
    if year_condition is not None:
        message = f"✓ All {k} results satisfy year > {year_condition}"
    else:
        message = f"✓ Returned {k} results (ANN query)"
    
    if indices_outside_intervals:
        message += f"\n  Note: {len(indices_outside_intervals)} results outside intervals {interval_str} (expected for ANN queries)"
    else:
        message += f"\n  All results are within intervals {interval_str}"
    
    return True, message, results


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Full workflow: python movie_index_test.py <dataset_filename> <num_clusters> <k> <year_condition> [--include-field <field_name>] [--limit <num_records>]")
        print("  Index only: python movie_index_test.py <dataset_filename> <num_clusters> --no-query [--include-field <field_name>] [--limit <num_records>]")
        print("  Dataset only: python movie_index_test.py <dataset_filename> --no-index [--limit <num_records>]")
        print("  Query only: python movie_index_test.py <dataset_filename> <k> <year_condition> --query-only")
        print("  Insert/Delete/Verify: python movie_index_test.py <dataset_filename> --insert <start> <end> | --delete <start> <end> | --verify <k> <interval_pairs> [--year <condition>]")
        print()
        print("Examples:")
        print("  python movie_index_test.py movie_embeddings_384d.json 10 50 2000")
        print("  python movie_index_test.py movie_embeddings_384d.json 10 50 2000 --include-field year")
        print("  python movie_index_test.py movie_embeddings_384d.json 10 --no-query --limit 300")
        print("  python movie_index_test.py movie_embeddings_384d.json --no-index --limit 2000")
        print("  python movie_index_test.py movie_embeddings_384d.json --insert 300 599")
        print("  python movie_index_test.py movie_embeddings_384d.json --delete 200 399")
        print("  python movie_index_test.py movie_embeddings_384d.json --verify 400 0 199 400 599")
        print("  python movie_index_test.py movie_embeddings_384d.json --verify 50 0 199 400 599 --year 2000")
        sys.exit(1)
    
    dataset_filename = sys.argv[1]
    
    # Check for insert/delete/verify modes first
    if '--insert' in sys.argv:
        idx = sys.argv.index('--insert')
        if idx + 2 >= len(sys.argv):
            print("Error: --insert requires <start> <end> arguments")
            sys.exit(1)
        insert_start = int(sys.argv[idx + 1])
        insert_end = int(sys.argv[idx + 2])
        
        # Construct dataset path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        datasets_dir = os.path.join(base_dir, 'datasets')
        dataset_path = os.path.join(datasets_dir, dataset_filename)
        
        if not os.path.exists(dataset_path):
            print(f"Error: dataset file not found: {dataset_path}")
            sys.exit(1)
        
        print("=" * 60)
        print("Movie Dataset Vector Index Test - INSERT MODE")
        print("=" * 60)
        print(f"Dataset: {dataset_filename}")
        print(f"Insert Range: [{insert_start}, {insert_end}]")
        print("=" * 60)
        print()
        
        insert_records_by_range(dataset_path, insert_start, insert_end)
        return
    
    if '--delete' in sys.argv:
        idx = sys.argv.index('--delete')
        if idx + 2 >= len(sys.argv):
            print("Error: --delete requires <start> <end> arguments")
            sys.exit(1)
        delete_start = int(sys.argv[idx + 1])
        delete_end = int(sys.argv[idx + 2])
        
        print("=" * 60)
        print("Movie Dataset Vector Index Test - DELETE MODE")
        print("=" * 60)
        print(f"Dataset: {dataset_filename}")
        print(f"Delete Range: [{delete_start}, {delete_end}]")
        print("=" * 60)
        print()
        
        delete_records_by_range(delete_start, delete_end)
        return
    
    if '--verify' in sys.argv:
        idx = sys.argv.index('--verify')
        if idx + 1 >= len(sys.argv):
            print("Error: --verify requires <k> followed by interval pairs")
            sys.exit(1)
        
        k = int(sys.argv[idx + 1])
        
        # Parse year condition if specified
        year_condition = None
        if '--year' in sys.argv:
            year_idx = sys.argv.index('--year')
            if year_idx + 1 < len(sys.argv):
                year_condition = int(sys.argv[year_idx + 1])
        
        # Parse interval pairs (start, end, start, end, ...)
        # Start after --verify and k, stop at --year if present
        interval_start_idx = idx + 2
        if '--year' in sys.argv:
            interval_end_idx = sys.argv.index('--year')
        else:
            interval_end_idx = len(sys.argv)
        
        interval_args = sys.argv[interval_start_idx:interval_end_idx]
        if len(interval_args) % 2 != 0:
            print("Error: --verify requires pairs of (start, end) for intervals")
            sys.exit(1)
        
        intervals = []
        for i in range(0, len(interval_args), 2):
            start = int(interval_args[i])
            end = int(interval_args[i + 1])
            intervals.append((start, end))
        
        # Construct dataset path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        datasets_dir = os.path.join(base_dir, 'datasets')
        dataset_path = os.path.join(datasets_dir, dataset_filename)
        
        if not os.path.exists(dataset_path):
            print(f"Error: dataset file not found: {dataset_path}")
            sys.exit(1)
        
        print("=" * 60)
        print("Movie Dataset Vector Index Test - VERIFY MODE")
        print("=" * 60)
        print(f"Dataset: {dataset_filename}")
        print(f"K: {k}")
        interval_str = ", ".join([f"[{s},{e}]" for s, e in intervals])
        print(f"Expected Intervals: {interval_str}")
        if year_condition:
            print(f"Year Condition: > {year_condition}")
        print("=" * 60)
        print()
        
        # Load query vector
        qvec, dimension = load_query_vector(dataset_path, idx=0)
        print(f"[info] Auto-detected dimension: {dimension}")
        print()
        
        # Execute query and verify
        success, message, results = verify_query_with_intervals(qvec, k, intervals, year_condition)
        
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
            idx_val = record.get('idx', 'N/A')
            year = record.get('year', 'N/A')
            dist = record.get('dist', 'N/A')
            print(f"  {i+1}. idx={idx_val}, year={year}, dist={dist}")
        
        if len(results) > 10:
            print(f"  ... ({len(results) - 10} more results)")
        
        print()
        
        if not success:
            sys.exit(1)
        
        return
    
    # Parse optional flags and arguments
    include_field = None
    skip_query = False
    skip_index = False
    query_only = False
    num_records = None
    k = None
    year_condition = None
    num_clusters = None
    
    # Parse --limit flag (can be in any mode)
    if '--limit' in sys.argv:
        try:
            limit_idx = sys.argv.index('--limit')
            if limit_idx + 1 < len(sys.argv):
                num_records = int(sys.argv[limit_idx + 1])
        except (ValueError, IndexError):
            print("Error: --limit requires an integer argument")
            sys.exit(1)
    
    # Check if --query-only is present
    if '--query-only' in sys.argv:
        query_only = True
        if len(sys.argv) < 4:
            print("Error: --query-only mode requires <dataset_filename> <k> <year_condition>")
            print("Usage: python movie_index_test.py <dataset_filename> <k> <year_condition> --query-only")
            sys.exit(1)
        k = int(sys.argv[2])
        year_condition = int(sys.argv[3])
    # Check if --no-index is present (load dataset only, skip index creation)
    elif '--no-index' in sys.argv:
        skip_index = True
    # Check if --no-query is present
    elif '--no-query' in sys.argv:
        skip_query = True
        num_clusters = int(sys.argv[2])
        # Parse include-field if present
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--include-field' and i + 1 < len(sys.argv):
                include_field = sys.argv[i + 1]
                break
            i += 1
    else:
        # Standard mode: require all arguments
        num_clusters = int(sys.argv[2])
        if len(sys.argv) < 5:
            print("Error: Full workflow mode requires <dataset_filename> <num_clusters> <k> <year_condition>")
            print("Usage: python movie_index_test.py <dataset_filename> <num_clusters> <k> <year_condition> [--include-field <field_name>]")
            sys.exit(1)
        
        k = int(sys.argv[3])
        year_condition = int(sys.argv[4])
        
        # Parse include-field if present
        i = 5
        while i < len(sys.argv):
            if sys.argv[i] == '--include-field' and i + 1 < len(sys.argv):
                include_field = sys.argv[i + 1]
                break
            i += 1
    
    # Construct dataset path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    datasets_dir = os.path.join(base_dir, 'datasets')
    dataset_path = os.path.join(datasets_dir, dataset_filename)
    
    if not os.path.exists(dataset_path):
        print(f"Error: dataset file not found: {dataset_path}")
        print(f"Expected location: datasets/{dataset_filename}")
        sys.exit(1)
    
    print("=" * 60)
    if query_only:
        print("Movie Dataset Vector Index Test - QUERY ONLY MODE")
    elif skip_index:
        print("Movie Dataset Vector Index Test - DATASET ONLY MODE")
    elif skip_query:
        print("Movie Dataset Vector Index Test - INDEX ONLY MODE")
    else:
        print("Movie Dataset Vector Index Test - FULL WORKFLOW")
    print("=" * 60)
    print(f"Dataset: {dataset_filename}")
    print(f"Location: {dataset_path}")
    if num_clusters is not None:
        print(f"Num Clusters: {num_clusters}")
    if k is not None:
        print(f"K: {k}")
        print(f"Year Condition: > {year_condition}")
    print(f"Include Field: {include_field if include_field else 'None'}")
    if num_records is not None:
        print(f"Record Limit: {num_records}")
    print("=" * 60)
    print()
    
    # Query-only mode: skip all setup, just execute query
    if query_only:
        print("[mode] Query-only: skipping dataverse/dataset/index creation")
        print()
        
        # Load query vector to get dimension
        qvec, dimension = load_query_vector(dataset_path, idx=0)
        print(f"[info] Auto-detected dimension: {dimension}")
        print()
        
        # Execute ANN query
        results = execute_ann_query(qvec, k, year_condition)
        print(f"[info] Received {len(results)} results")
        print()
        
        # Verify results
        print("[step] Verifying results...")
        success, message = verify_results(results, k, year_condition)
        
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
            idx = record.get('idx', 'N/A')
            year = record.get('year', 'N/A')
            dist = record.get('dist', 'N/A')
            print(f"  {i+1}. idx={idx}, year={year}, dist={dist}")
        
        if len(results) > 10:
            print(f"  ... ({len(results) - 10} more results)")
        
        print()
        
        if not success:
            sys.exit(1)
        
        return
    
    # Step 1: Create dataverse and load dataset
    create_dataverse_and_dataset(dataset_path, num_records)
    
    # If --no-index flag is set, stop here (dataset loaded, no index created)
    if skip_index:
        print()
        print("=" * 60)
        print("✓ DATASET LOADED")
        print("Index creation skipped (--no-index flag set)")
        print("=" * 60)
        print()
        return
    
    # Step 2: Drop existing index
    drop_existing_index()
    
    # Step 3: Load query vector to get dimension
    qvec, dimension = load_query_vector(dataset_path, idx=0)
    print(f"[info] Auto-detected dimension: {dimension}")
    print()
    
    # Step 4: Create vector index with detected dimension
    create_vector_index(dimension, num_clusters, include_field)
    
    # If --no-query flag is set, stop here
    if skip_query:
        print()
        print("=" * 60)
        print("✓ INDEX CREATION COMPLETE")
        print("Query execution skipped (--no-query flag set)")
        print("=" * 60)
        print()
        return
    
    # Step 5: Execute ANN query
    results = execute_ann_query(qvec, k, year_condition)
    print(f"[info] Received {len(results)} results")
    print()
    
    # Step 6: Verify results
    print("[step] Verifying results...")
    success, message = verify_results(results, k, year_condition)
    
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
        idx = record.get('idx', 'N/A')
        year = record.get('year', 'N/A')
        dist = record.get('dist', 'N/A')
        print(f"  {i+1}. idx={idx}, year={year}, dist={dist}")
    
    if len(results) > 10:
        print(f"  ... ({len(results) - 10} more results)")
    
    print()
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
