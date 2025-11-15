# ANN Benchmark Pipeline

This repository provides a full end-to-end pipeline for running ANN (Approximate Nearest Neighbor) experiments using:

- **ANN-Benchmarks datasets** (in HDF5 format)
- A **custom vector index** built in **Apache AsterixDB**
- A suite of Python scripts to:
  1. Download datasets
  2. Convert HDF5 → JSONL
  3. Load data into AsterixDB
  4. Build vector indexes
  5. Run ANN queries + compute recall

All steps can be run either using one single **pipeline script** or using the individual scripts manually.

------

## 📂 Directory Structure

```
project_root/
  raw/                # downloaded .hdf5 files (from ann-benchmarks.com)
  datasets/           # train vectors in JSONL format
  tests/              # query vectors in JSONL format
  neighbors/          # ground-truth neighbors in JSONL format

  scripts/
    pipeline.py       # orchestrates full workflow
    hdf5_to_jsonl.py  # converts HDF5 → train/test/neighbors JSONL
    load_dataset.py   # loads JSONL into AsterixDB using localfs
    create_index.py   # creates vector index in AsterixDB
    run_query.py      # executes ANN queries + recall evaluation
```

------

## Prerequisites

- Python **3.8+**

- Install dependencies:

  ```
  pip install requests h5py numpy tqdm
  ```

- Apache **AsterixDB** running locally at:

  ```
  http://localhost:19002
  ```

------

# 1. Full Pipeline

Run the entire ANN workflow with one command:

```
python scripts/pipeline.py <dataset_name> <num_k> <num_queries>
```

### Example

```
python scripts/pipeline.py fashion-mnist-784-euclidean 256 1000
```

This performs:

1. **Download** HDF5 to `/raw`
2. **Convert** HDF5 → JSONL
   - `/datasets/<name>_train.jsonl`
   - `/tests/<name>_test.jsonl`
   - `/neighbors/<name>_neighbors.jsonl`
3. **Load** JSONL into AsterixDB (`VectorTest`)
4. **Create a Vector Index**
   - Dimension auto-inferred (e.g., 784)
5. **Run ANN queries**
   - Compute Recall@100 for each query

------

# 2. Clean Generated Files

Remove all files for a given dataset:

```
python scripts/pipeline.py <dataset_name> clean
```

Example:

```
python scripts/pipeline.py fashion-mnist-784-euclidean clean
```

This deletes:

- `raw/<dataset>.hdf5`
- `datasets/<dataset>_train.jsonl`
- `tests/<dataset>_test.jsonl`
- `neighbors/<dataset>_neighbors.jsonl`

------

# 3. Using Scripts Individually

Each stage can be run manually if needed.

------

## 3.1 Convert HDF5 → JSONL

```
python scripts/hdf5_to_jsonl.py <dataset_name>
```

Example:

```
python scripts/hdf5_to_jsonl.py fashion-mnist-784-euclidean
```

Produces:

- `datasets/<dataset>_train.jsonl`
- `tests/<dataset>_test.jsonl`
- `neighbors/<dataset>_neighbors.jsonl`

------

## 3.2 Load Dataset into AsterixDB

```
python scripts/load_dataset.py <dataset_name>
```

Example:

```
python scripts/load_dataset.py fashion-mnist-784-euclidean
```

This:

- Creates `DATAVERSE VectorTest`

- Creates dataset `VectorTest.<dataset_name_with_underscores>`

- Loads JSONL using:

  ```sql
  LOAD DATASET ... USING localfs
  ```

------

## 3.3 Create Vector Index

```
python scripts/create_index.py <dataset_name> <num_k>
```

Example:

```
python scripts/create_index.py fashion-mnist-784-euclidean 256
```

Automatically:

- Converts `-` to `_`

- Extracts vector dimension from the name (e.g. `784`)

- Sets `"similarity": "Euclidean"` or `"Angular"` based on dataset name

- Creates:

  ```sql
  CREATE VECTOR INDEX ix ON dataset(embedding)
  ```

------

## 3.4 Run Queries + Compute Recall

```
python scripts/run_query.py <dataset_name> <num_queries>
```

Example:

```
python scripts/run_query.py fashion-mnist-784-euclidean 500
```

The script:

- Loads queries from:

  ```
  tests/<dataset>_test.jsonl
  ```

- Loads ground truth from:

  ```
  neighbors/<dataset>_neighbors.jsonl
  ```

- Sends ANN query to AsterixDB:

  ```sql
  SELECT row.idx
  FROM dataset AS row
  ORDER BY ann_distance(row.embedding, target, "Euclidean")
  LIMIT 100;
  ```

- Computes Recall@100 for each query

- Prints per-query recall + final mean recall

