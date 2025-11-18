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
  raw/                     # downloaded .hdf5 files (from ann-benchmarks.com)
  datasets/                # train vectors in JSONL format
                           # includes subdatasets: <name>_train_<N>.jsonl
  tests/                   # query vectors in JSONL format
  neighbors/               # ground-truth neighbors in JSONL format
  output/                  # timestamped result files from benchmarks

  scripts/
    pipeline.py            # orchestrates full workflow
    hdf5_to_jsonl.py       # converts HDF5 → train/test/neighbors JSONL
    create_subdataset.py   # creates subdatasets with N records
    load_dataset.py        # loads JSONL into AsterixDB using localfs
    create_index.py        # creates vector index in AsterixDB
    run_query.py           # executes ANN queries vs pre-computed ground truth
    run_query_compare.py   # compares ANN vs exact distance (for subdatasets)
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
python scripts/pipeline.py <dataset_name> <num_k> <num_queries> [num_records]
```

### Example (Full Dataset)

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
   - Compares ANN (`ann_distance`) vs Exact (`vector_distance`)
   - Computes Recall@100 for each query

### Example (Subdataset)

To work with a smaller subset of the training data (e.g., 20,000 records instead of 60,000):

```
python scripts/pipeline.py fashion-mnist-784-euclidean 256 1000 20000
```

This adds an extra step:

2.5. **Create Subdataset**
   - Extracts first 20,000 records from `<name>_train.jsonl`
   - Creates `<name>_train_20000.jsonl`
   - Loads into AsterixDB as `<name>_20000`
   - Creates index on the subdataset
   - Runs queries against the subdataset

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
- `datasets/<dataset>_train_*.jsonl` (all subdatasets)
- `tests/<dataset>_test.jsonl`
- `neighbors/<dataset>_neighbors.jsonl`

------

# 3. Working with Subdatasets

When working with large datasets (e.g., 60,000+ records), you may want to test with smaller subsets for faster iteration and experimentation.

## Why Use Subdatasets?

- **Faster loading** into AsterixDB
- **Quicker index creation**
- **Faster query evaluation**
- **Easier testing and debugging**
- **Accurate recall metrics** using actual database content

## How It Works

1. Specify the number of records as the 4th parameter to `pipeline.py`
2. The pipeline creates a new file: `<dataset>_train_<N>.jsonl`
3. This file contains the first N records from the full training set
4. A new AsterixDB dataset is created: `<dataset>_<N>`
5. Queries compare ANN vs exact distance on the subdataset

## Example

```bash
# Full dataset (60,000 records)
python scripts/pipeline.py fashion-mnist-784-euclidean 256 1000

# Subdataset (20,000 records)
python scripts/pipeline.py fashion-mnist-784-euclidean 256 1000 20000

# Subdataset (10,000 records)
python scripts/pipeline.py fashion-mnist-784-euclidean 256 1000 10000
```

## File Naming Convention

| Full Dataset | Subdataset (20k) |
|-------------|------------------|
| `fashion-mnist-784-euclidean_train.jsonl` | `fashion-mnist-784-euclidean_train_20000.jsonl` |
| AsterixDB: `fashion_mnist_784_euclidean` | AsterixDB: `fashion_mnist_784_euclidean_20000` |

## Recall Calculation for Subdatasets

For subdatasets, recall is computed by:

1. Running an **ANN query** using `ann_distance()` (uses the vector index)
2. Running an **exact query** using `vector_distance()` (brute-force search)
3. Comparing the two result sets to calculate recall@K

This ensures accurate recall metrics that reflect the actual data in your database, without relying on pre-computed ground truth files.

------

# 4. Using Scripts Individually

Each stage can be run manually if needed.

------

## 4.1 Convert HDF5 → JSONL

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

## 4.2 Create Subdataset

```
python scripts/create_subdataset.py <dataset_name> <num_records>
```

Example:

```
python scripts/create_subdataset.py fashion-mnist-784-euclidean 20000
```

This:

- Reads the first `num_records` from `datasets/<dataset>_train.jsonl`
- Creates `datasets/<dataset>_train_<num_records>.jsonl`
- Useful for creating multiple subdatasets of different sizes

------

## 4.3 Load Dataset into AsterixDB

```
python scripts/load_dataset.py <dataset_name> [num_records]
```

Example (full dataset):

```
python scripts/load_dataset.py fashion-mnist-784-euclidean
```

Example (subdataset):

```
python scripts/load_dataset.py fashion-mnist-784-euclidean 20000
```

This:

- Creates `DATAVERSE VectorTest`

- Creates dataset `VectorTest.<dataset_name_with_underscores>`
  - If `num_records` is specified, dataset name becomes `<dataset>_<num_records>`

- Loads JSONL using:

  ```sql
  LOAD DATASET ... USING localfs
  ```

------

## 4.4 Create Vector Index

```
python scripts/create_index.py <dataset_name> <num_k> [num_records]
```

Example (full dataset):

```
python scripts/create_index.py fashion-mnist-784-euclidean 256
```

Example (subdataset):

```
python scripts/create_index.py fashion-mnist-784-euclidean 256 20000
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

## 4.5 Run Queries + Compute Recall

### Option A: Compare with Pre-computed Ground Truth

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

### Option B: Compare ANN vs Exact Distance (Recommended for Subdatasets)

```
python scripts/run_query_compare.py <dataset_name> <num_queries> [num_records]
```

Example (full dataset):

```
python scripts/run_query_compare.py fashion-mnist-784-euclidean 500
```

Example (subdataset):

```
python scripts/run_query_compare.py fashion-mnist-784-euclidean 500 20000
```

The script:

- For each test query, runs **two queries**:
  
  1. **ANN Query** (approximate):
     ```sql
     ORDER BY ann_distance(row.embedding, target, "Euclidean")
     ```
  
  2. **Exact Query** (ground truth from actual DB):
     ```sql
     ORDER BY vector_distance(row.embedding, target, "Euclidean")
     ```

- Computes recall by comparing ANN results against exact results

- **Tracks execution time statistics** for both query types

- Displays comprehensive performance metrics:
  - Per-query recall and timing
  - Average query times for ANN and exact queries
  - Speedup factor (how much faster ANN is vs exact)
  - Total execution times

- **Automatically saves results** to timestamped files in `output/` directory

- **Advantages**: 
  - Works with subdatasets without needing pre-computed ground truth
  - Provides quantitative performance comparison
  - Persistent records for historical tracking

**Sample output:**
```
Query 0: Recall@100 = 0.9800 | ANN: 0.005234s | Exact: 0.045678s
...
==============================================
RESULTS SUMMARY
==============================================
Queries evaluated:        1000
Mean Recall@100:          0.9723

Avg ANN query time:       0.005124s
Avg Exact query time:     0.045234s
Speedup (Exact/ANN):      8.83x

Total ANN time:           5.124s
Total Exact time:         45.234s
==============================================

Results saved to: output/fashion-mnist-784-euclidean_20000_results_20231117_143052.txt
```

