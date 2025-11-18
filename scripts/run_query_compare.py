import json
import requests
import os
import sys
from datetime import datetime

# --------------------
# Config
# --------------------
ASTERIX_URL = "http://localhost:19002/query/service"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}

TOP_K = 100  # number of neighbors to retrieve per query


def load_test_vectors(path, limit=None):
    """Load query vectors from test.jsonl."""
    vectors = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            vectors.append(obj["embedding"])
            if limit is not None and len(vectors) >= limit:
                break
    return vectors


def build_ann_statement(target_vec, top_k, asterix_dataset_name):
    """Build ANN query using ann_distance."""
    target_literal = ", ".join(str(x) for x in target_vec)

    statement = f"""
    USE VectorTest;
    LET target=[{target_literal}]
    FROM {asterix_dataset_name} row
    LET dist = ann_distance(row.embedding, target, "Euclidean")
    SELECT row.idx
    ORDER BY dist
    LIMIT {top_k};
    """
    return statement


def build_exact_statement(target_vec, top_k, asterix_dataset_name):
    """Build exact query using vector_distance."""
    target_literal = ", ".join(str(x) for x in target_vec)

    statement = f"""
    USE VectorTest;
    LET target=[{target_literal}]
    FROM {asterix_dataset_name} row
    LET dist = vector_distance(row.embedding, target, "Euclidean")
    SELECT row.idx
    ORDER BY dist
    LIMIT {top_k};
    """
    return statement


def execute_query(statement):
    """Send a query to AsterixDB and return results and execution time."""
    data = {
        "statement": statement,
        "pretty": "false",
        "client_context_id": "ann_eval"
    }
    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    resp.raise_for_status()
    js = resp.json()
    rows = js.get("results", [])

    ids = []
    for row in rows:
        if "idx" in row:
            ids.append(row["idx"])
        elif "row.idx" in row:
            ids.append(row["row.idx"])
        else:
            raise ValueError(f"Unexpected row format: {row}")
    
    # Extract execution time from metrics
    metrics = js.get("metrics", {})
    execution_time_str = metrics.get("executionTime", "0s")
    
    # Parse execution time (formats: "2.871s", "115.961ms", "1.5m", "500ns")
    execution_time = parse_time_to_seconds(execution_time_str)
    
    return ids, execution_time


def parse_time_to_seconds(time_str):
    """
    Parse time string from AsterixDB metrics to seconds.
    Supports formats: "2.871s", "115.961ms", "1.5m", "500ns", "500µs"
    """
    time_str = time_str.strip()
    
    if time_str.endswith('ns'):
        # nanoseconds
        return float(time_str.rstrip('ns')) / 1_000_000_000
    elif time_str.endswith('µs') or time_str.endswith('us'):
        # microseconds
        value = time_str.rstrip('µs').rstrip('us')
        return float(value) / 1_000_000
    elif time_str.endswith('ms'):
        # milliseconds
        return float(time_str.rstrip('ms')) / 1_000
    elif time_str.endswith('s'):
        # seconds
        return float(time_str.rstrip('s'))
    elif time_str.endswith('m'):
        # minutes
        return float(time_str.rstrip('m')) * 60
    elif time_str.endswith('h'):
        # hours
        return float(time_str.rstrip('h')) * 3600
    else:
        # default to seconds if no unit
        return float(time_str)


def calculate_recall(ann_results, exact_results):
    """Calculate recall: how many ANN results are in the exact results."""
    exact_set = set(exact_results)
    hits = sum(1 for id_ in ann_results if id_ in exact_set)
    recall = hits / len(exact_results) if exact_results else 0.0
    return recall


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python run_query_compare.py <dataset_name> <num_queries> [num_records]")
        print("Example: python run_query_compare.py fashion-mnist-784-euclidean 1000")
        print("Example: python run_query_compare.py fashion-mnist-784-euclidean 1000 20000")
        sys.exit(1)

    dataset_name = sys.argv[1]
    num_queries = int(sys.argv[2])
    num_records = sys.argv[3] if len(sys.argv) == 4 else None

    # Adjust dataset name for subdataset
    if num_records:
        ds_name_astx = f"{dataset_name}_{num_records}".replace("-", "_")
        display_name = f"{dataset_name} (subdataset: {num_records} records)"
    else:
        ds_name_astx = dataset_name.replace("-", "_")
        display_name = dataset_name

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_path = os.path.join(base_dir, "tests", f"{dataset_name}_test.jsonl")

    if not os.path.exists(tests_path):
        print(f"Error: test file not found: {tests_path}")
        sys.exit(1)

    # Prepare output directory and file
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if num_records:
        output_filename = f"{dataset_name}_{num_records}_results_{timestamp}.txt"
    else:
        output_filename = f"{dataset_name}_results_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Open output file for writing
    output_file = open(output_path, "w")
    
    def tee_print(msg):
        """Print to both console and file."""
        print(msg)
        output_file.write(msg + "\n")
        output_file.flush()

    tee_print("==============================================")
    tee_print(f"Dataset:            {display_name}")
    tee_print(f"Asterix dataset:    {ds_name_astx}")
    tee_print(f"Queries to evaluate:{num_queries}")
    tee_print(f"Comparing ANN (ann_distance) vs Exact (vector_distance)")
    tee_print("==============================================\n")

    tee_print("Loading query vectors...")
    test_vecs = load_test_vectors(tests_path, limit=num_queries)
    tee_print(f"Loaded {len(test_vecs)} query vectors\n")

    total_recall = 0.0
    total_ann_time = 0.0
    total_exact_time = 0.0

    for qid, vec in enumerate(test_vecs):
        # Run ANN query
        ann_ids, ann_time = execute_query(build_ann_statement(vec, TOP_K, ds_name_astx))
        total_ann_time += ann_time
        
        # Run exact query
        exact_ids, exact_time = execute_query(build_exact_statement(vec, TOP_K, ds_name_astx))
        total_exact_time += exact_time
        
        # Calculate recall
        recall = calculate_recall(ann_ids, exact_ids)
        total_recall += recall

        tee_print(f"Query {qid}: Recall@{TOP_K} = {recall:.4f} | "
                  f"ANN: {ann_time:.6f}s | Exact: {exact_time:.6f}s")

        if (qid + 1) % 50 == 0:
            mean_recall = total_recall / (qid + 1)
            mean_ann_time = total_ann_time / (qid + 1)
            mean_exact_time = total_exact_time / (qid + 1)
            tee_print(f"Processed {qid + 1}/{num_queries} queries. "
                      f"Mean Recall@{TOP_K} = {mean_recall:.4f} | "
                      f"Avg ANN time: {mean_ann_time:.6f}s | "
                      f"Avg Exact time: {mean_exact_time:.6f}s")

    mean_recall = total_recall / num_queries
    mean_ann_time = total_ann_time / num_queries
    mean_exact_time = total_exact_time / num_queries
    speedup = mean_exact_time / mean_ann_time if mean_ann_time > 0 else 0
    
    tee_print("\n==============================================")
    tee_print(f"RESULTS SUMMARY")
    tee_print("==============================================")
    tee_print(f"Queries evaluated:        {num_queries}")
    tee_print(f"Mean Recall@{TOP_K}:        {mean_recall:.4f}")
    tee_print(f"")
    tee_print(f"Avg ANN query time:       {mean_ann_time:.6f}s")
    tee_print(f"Avg Exact query time:     {mean_exact_time:.6f}s")
    tee_print(f"Speedup (Exact/ANN):      {speedup:.2f}x")
    tee_print(f"")
    tee_print(f"Total ANN time:           {total_ann_time:.3f}s")
    tee_print(f"Total Exact time:         {total_exact_time:.3f}s")
    tee_print("==============================================\n")
    
    # Close output file
    output_file.close()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
