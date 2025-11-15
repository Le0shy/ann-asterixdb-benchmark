import os
import sys
import requests
import re

ASTERIX_URL = "http://localhost:19002/query/service"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def extract_dimension(dataset_name: str) -> int:
    """
    Extracts the dimension from dataset_name.
    Examples:
        fashion-mnist-784-euclidean -> 784
        glove-200-angular -> 200
    """
    parts = dataset_name.split("-")
    for p in parts:
        if p.isdigit():              # take the first pure integer token
            return int(p)
    raise ValueError(f"Cannot infer dimension from dataset name: {dataset_name}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python create_index.py <dataset_name> <num_k>")
        print("Example: python create_index.py fashion-mnist-784-euclidean 256")
        sys.exit(1)

    dataset_name = sys.argv[1]                     # e.g. fashion-mnist-784-euclidean
    ds_name_astx = dataset_name.replace("-", "_")  # Asterix-safe name

    num_k = int(sys.argv[2])                       # number of leaf centroids

    # Automatically extract dimension
    dimension = extract_dimension(dataset_name)

    index_name = "ix"

    statement = f"""
    USE VectorTest;

    DROP INDEX {ds_name_astx}.{index_name} IF EXISTS;

    CREATE VECTOR INDEX {index_name} ON {ds_name_astx}(embedding VECTOR) WITH {{
        "dimension": {dimension},
        "train_list": 10000,
        "description": " ",
        "num_k": {num_k},
        "similarity": "Euclidean"
    }};
    """

    data = {
        "statement": statement,
        "pretty": "true",
        "client_context_id": f"create_idx_{dataset_name}"
    }

    print("Creating vector index:")
    print(f"  Dataset:        {ds_name_astx}")
    print(f"  Auto-dimension: {dimension}")
    print(f"  num_k:          {num_k}")
    print()

    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("HTTP error from AsterixDB:", e)
        print("Response text:")
        print(resp.text)
        sys.exit(1)

    print("AsterixDB response:")
    print(resp.text)


if __name__ == "__main__":
    main()
