import os
import sys
import requests

ASTERIX_URL = "http://localhost:19002/query/service"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python load_dataset.py <dataset_name> [num_records]")
        print("Example: python load_dataset.py fashion-mnist-784-euclidean")
        print("Example: python load_dataset.py fashion-mnist-784-euclidean 20000")
        sys.exit(1)

    dataset_name = sys.argv[1]                    # original name (with hyphens)
    num_records = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Adjust dataset name for subdataset
    if num_records:
        dataset_file_name = f"{dataset_name}_train_{num_records}.jsonl"
        ds_name_astx = f"{dataset_name}_{num_records}".replace("-", "_")
    else:
        dataset_file_name = f"{dataset_name}_train.jsonl"
        ds_name_astx = dataset_name.replace("-", "_") # Asterix-friendly dataset name

    # Project paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, "datasets")

    # JSONL file on disk
    json_path = os.path.join(datasets_dir, dataset_file_name)
    if not os.path.exists(json_path):
        print(f"Error: dataset file not found: {json_path}")
        sys.exit(1)

    # Normalize to forward slashes
    json_path_fs = json_path.replace("\\", "/")

    # localfs expects: Host://AbsolutePath
    # e.g. "127.0.0.1:///Users/hongyu/Projects/..."
    host = "localhost"
    json_path_for_asterix = f"{host}://{json_path_fs}"

    statement = f"""
    DROP DATAVERSE VectorTest IF EXISTS;
    CREATE DATAVERSE VectorTest;
    USE VectorTest;

    CREATE TYPE OpenType AS {{
      idx: int
    }};

    CREATE DATASET {ds_name_astx} (OpenType)
    PRIMARY KEY idx WITH {{
      "storage-format": {{"format":"column"}}
    }};

    USE VectorTest;

    LOAD DATASET {ds_name_astx} USING localfs (
      ("path" = "{json_path_for_asterix}"),
      ("format" = "json")
    );
    """

    data = {
        "statement": statement,
        "pretty": "true",
        "client_context_id": f"load_{dataset_name}"
    }

    print(f"Loading dataset '{dataset_name}' as Asterix dataset '{ds_name_astx}'")
    print(f"Path for localfs: {json_path_for_asterix}\n")

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


