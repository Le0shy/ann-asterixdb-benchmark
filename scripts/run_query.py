import json
import requests
import os
import sys

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


def load_ground_truth(path, limit=None, k_limit=None):
    """Load ground-truth neighbor ids from neighbors.jsonl."""
    gts = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            neigh = obj["neighbors"]
            if k_limit is not None and len(neigh) > k_limit:
                neigh = neigh[:k_limit]
            gts.append(neigh)
            if limit is not None and len(gts) >= limit:
                break
    return gts


def build_statement(target_vec, top_k, asterix_dataset_name):
    """Convert target vector into SQL++ literal and build the ANN query."""
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


def get_ann_results(target_vec, top_k, asterix_dataset_name):
    """Send a query to AsterixDB and return ANN results."""
    statement = build_statement(target_vec, top_k, asterix_dataset_name)
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
    return ids


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_recall_dynamic.py <dataset_name> <num_queries>")
        print("Example: python run_recall_dynamic.py fashion-mnist-784-euclidean 1000")
        sys.exit(1)

    dataset_name = sys.argv[1]                     # e.g. "fashion-mnist-784-euclidean"
    num_queries = int(sys.argv[2])                 # dynamic number of queries

    ds_name_astx = dataset_name.replace("-", "_")  # Asterix-friendly dataset name

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_path = os.path.join(base_dir, "tests", f"{dataset_name}_test.jsonl")
    neighbors_path = os.path.join(base_dir, "neighbors", f"{dataset_name}_neighbors.jsonl")

    if not os.path.exists(tests_path):
        print(f"Error: test file not found: {tests_path}")
        sys.exit(1)
    if not os.path.exists(neighbors_path):
        print(f"Error: neighbors file not found: {neighbors_path}")
        sys.exit(1)

    print(f"Dataset:            {dataset_name}")
    print(f"Asterix dataset:    {ds_name_astx}")
    print(f"Queries to evaluate:{num_queries}")
    print()

    print("Loading query vectors and ground-truth neighbors...")
    test_vecs = load_test_vectors(tests_path, limit=num_queries)
    gt_lists = load_ground_truth(neighbors_path, limit=num_queries, k_limit=TOP_K)

    assert len(test_vecs) == len(gt_lists), "Mismatch between test and neighbor lengths"

    total_recall = 0.0

    for qid, (vec, gt) in enumerate(zip(test_vecs, gt_lists)):
        ann_ids = get_ann_results(vec, TOP_K, ds_name_astx)
        gt_set = set(gt)

        hit = sum(1 for id_ in ann_ids if id_ in gt_set)
        recall = hit / len(gt_set) if gt_set else 0.0
        total_recall += recall

        print(f"Query {qid}: Recall@{TOP_K} = {recall:.4f}")

        if (qid + 1) % 50 == 0:
            print(f"Processed {qid + 1}/{num_queries} queries. "
                  f"Mean Recall@{TOP_K} = {total_recall / (qid + 1):.4f}")

    mean_recall = total_recall / num_queries
    print("\n==============================================")
    print(f"Final mean Recall@{TOP_K} over {num_queries} queries = {mean_recall:.4f}")
    print("==============================================\n")


if __name__ == "__main__":
    main()

