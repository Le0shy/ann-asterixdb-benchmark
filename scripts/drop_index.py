#!/usr/bin/env python3
"""
Drop a vector index without dropping the dataset.

Usage:
    python drop_index.py <dataset_name> [num_records]

Examples:
    python drop_index.py glove-100-angular
    python drop_index.py fashion-mnist-784-euclidean 20000
"""

import os
import sys
import requests

ASTERIX_URL = "http://localhost:19002/query/service"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python drop_index.py <dataset_name> [num_records]")
        print()
        print("Examples:")
        print("  python drop_index.py glove-100-angular")
        print("  python drop_index.py fashion-mnist-784-euclidean 20000")
        sys.exit(1)

    dataset_name = sys.argv[1]
    num_records = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Adjust dataset name for subdataset
    if num_records:
        ds_name_astx = f"{dataset_name}_{num_records}".replace("-", "_")
    else:
        ds_name_astx = dataset_name.replace("-", "_")

    index_name = "ix1"

    statement = f"""
    USE VectorTest;
    DROP INDEX {ds_name_astx}.{index_name} IF EXISTS;
    """

    data = {
        "statement": statement,
        "pretty": "true",
        "client_context_id": f"drop_idx_{dataset_name}"
    }

    print(f"Dropping index '{index_name}' from dataset '{ds_name_astx}'...")
    print()

    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("HTTP error from AsterixDB:", e)
        print("Response text:")
        print(resp.text)
        sys.exit(1)

    print("✓ Index dropped successfully")
    print()
    print("AsterixDB response:")
    print(resp.text)


if __name__ == "__main__":
    main()
