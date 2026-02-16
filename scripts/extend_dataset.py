import os
import sys
import subprocess
import requests
import json
from datetime import datetime

ASTERIX_URL = "http://localhost:19002/query/service"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def run_subprocess(cmd, cwd=None):
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[run] FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)


def make_localfs_path(path: str) -> str:
    # Normalize and convert to localfs host format: localhost:///absolute/path
    p = os.path.abspath(path).replace('\\', '/')
    host = 'localhost'
    return f"{host}://{p}"


def load_insert_file_into_asterix(dataset_astx: str, insert_file_path: str):
    # Build an INSERT INTO statement by loading the JSONL file and creating
    # a JSON array literal. This uses the AsterixDB INSERT INTO (<array>) syntax.
    if not os.path.exists(insert_file_path):
        print(f"Error: insert file not found: {insert_file_path}")
        sys.exit(1)

    # Read file and collect JSON objects (one per line)
    objs = []
    with open(insert_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(line)

    if not objs:
        print('No records to insert.')
        return

    # Build array literal string
    array_literal = ',\n'.join(objs)

    statement = f"""
    USE VectorTest;
    INSERT INTO {dataset_astx} ([
    {array_literal}
    ]);
    """

    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'extend_dataset_insert'
    }

    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print('HTTP error from AsterixDB during insert:', e)
        print('Response text:')
        print(resp.text)
        sys.exit(1)
    print('AsterixDB response for insert:')
    print(resp.text)


def load_upsert_file_into_asterix(dataset_astx: str, upsert_file_path: str):
    """
    Build an UPSERT INTO statement by loading the JSONL file and creating
    a JSON array literal. This uses the AsterixDB UPSERT INTO (<array>) syntax.
    UPSERT will insert new records or update existing ones based on primary key.
    """
    if not os.path.exists(upsert_file_path):
        print(f"Error: upsert file not found: {upsert_file_path}")
        sys.exit(1)

    # Read file and collect JSON objects (one per line)
    objs = []
    with open(upsert_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(line)

    if not objs:
        print('No records to upsert.')
        return

    # Build array literal string
    array_literal = ',\n'.join(objs)

    statement = f"""
    USE VectorTest;
    UPSERT INTO {dataset_astx} ([
    {array_literal}
    ]);
    """

    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'extend_dataset_upsert'
    }

    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print('HTTP error from AsterixDB during upsert:', e)
        print('Response text:')
        print(resp.text)
        sys.exit(1)
    print('AsterixDB response for upsert:')
    print(resp.text)


def create_insert_file(full_train_path: str, start: int, count: int, out_path: str):
    if not os.path.exists(full_train_path):
        print(f"Error: full train file not found: {full_train_path}")
        sys.exit(1)
    read = 0
    written = 0
    with open(full_train_path, 'r') as inf, open(out_path, 'w') as outf:
        for line in inf:
            if read < start:
                read += 1
                continue
            if written >= count:
                break
            outf.write(line)
            written += 1
    return written


def create_insert_file_by_index(full_train_path: str, idx_start: int, idx_end: int, out_path: str):
    """
    Create an insert file by selecting records whose "idx" field is within [idx_start, idx_end].
    Returns the number of records written.
    """
    if not os.path.exists(full_train_path):
        print(f"Error: full train file not found: {full_train_path}")
        sys.exit(1)

    written = 0
    with open(full_train_path, 'r') as inf, open(out_path, 'w') as outf:
        for line in inf:
            line_strip = line.strip()
            if not line_strip:
                continue
            try:
                obj = json.loads(line_strip)
            except Exception:
                # If parsing fails, skip line
                continue
            idx = obj.get('idx')
            if idx is None:
                continue
            if idx_start <= idx <= idx_end:
                outf.write(line)
                written += 1
            # early exit if passed range
            if idx > idx_end:
                break
    return written


# New helper: check whether the target dataset exists in AsterixDB
def dataset_exists_in_asterix(dataset_astx: str) -> bool:
    """Return True if a simple COUNT query against the dataset succeeds."""
    stmt = f"USE VectorTest; SELECT count(*) FROM {dataset_astx};"
    data = {
        'statement': stmt,
        'pretty': 'true',
        'client_context_id': 'extend_dataset_check'
    }
    try:
        resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data, timeout=10)
        if resp.status_code != 200:
            return False
        text = resp.text.lower()
        # Asterix returns errors inline; a quick heuristic: if 'exception' or 'error' appears, consider it missing
        if 'exception' in text or 'error' in text:
            return False
        return True
    except Exception:
        return False


def batch_delete_records(dataset_astx: str, idx_start: int, idx_end: int):
    """
    Delete records from AsterixDB dataset where idx is in [idx_start, idx_end].
    Uses DELETE FROM ... WHERE r.idx >= start AND r.idx <= end syntax.
    """
    statement = f"""
    USE VectorTest;
    DELETE FROM {dataset_astx} r WHERE r.idx >= {idx_start} AND r.idx <= {idx_end};
    """
    
    data = {
        'statement': statement,
        'pretty': 'true',
        'client_context_id': 'extend_dataset_delete'
    }
    
    print(f'[step] Deleting records with idx in [{idx_start}, {idx_end}] from {dataset_astx}')
    resp = requests.post(ASTERIX_URL, headers=HEADERS, data=data)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print('HTTP error from AsterixDB during delete:', e)
        print('Response text:')
        print(resp.text)
        sys.exit(1)
    print('AsterixDB response for delete:')
    print(resp.text)


def main():
    # Extended CLI parsing to support four modes:
    # 1) insert_count:  python extend_dataset.py <dataset> <num_clusters> <initial_records> <num_inserted>
    # 2) insert_range:  python extend_dataset.py <dataset> <num_clusters> <initial_records> <start_idx> <end_idx>
    # 3) delete_range:  python extend_dataset.py <dataset> <num_clusters> <initial_records> --delete-range <start_idx> <end_idx>
    # 4) upsert_range:  python extend_dataset.py <dataset> <num_clusters> <initial_records> --upsert-range <start_idx> <end_idx>
    # Backwards-compatible: a negative 4th positional arg still means delete from 0..abs(arg)

    # Basic sanity check for argument count (allowing the new flag form)
    if len(sys.argv) < 4:
        print('Usage (insert count):   python extend_dataset.py <dataset_name> <num_clusters> <initial_records> <num_inserted>')
        print('Usage (insert range):   python extend_dataset.py <dataset_name> <num_clusters> <initial_records> <start_idx> <end_idx>')
        print('Usage (delete range):   python extend_dataset.py <dataset_name> <num_clusters> <initial_records> --delete-range <start_idx> <end_idx>')
        print('Usage (upsert range):   python extend_dataset.py <dataset_name> <num_clusters> <initial_records> --upsert-range <start_idx> <end_idx>')
        print('Example (insert count): python extend_dataset.py gist-960-euclidean 20 10000 5000')
        print('Example (insert range): python extend_dataset.py gist-960-euclidean 20 10000 15000 20000')
        print('Example (delete, compat): python extend_dataset.py gist-960-euclidean 20 600 -299')
        print('Example (delete range): python extend_dataset.py gist-960-euclidean 20 600 --delete-range 0 299')
        print('Example (upsert range): python extend_dataset.py gist-960-euclidean 20 300 --upsert-range 200 399')
        sys.exit(1)

    # Common args
    dataset_name = sys.argv[1]
    num_clusters = sys.argv[2]
    
    # Check if arg3 is a flag (--delete-range or --upsert-range) - special case for when initial_records is omitted
    # This shouldn't normally happen but let's handle it gracefully
    if sys.argv[3].startswith('--'):
        print(f"Error: Missing initial_records argument before {sys.argv[3]}")
        print("Correct format: python extend_dataset.py <dataset_name> <num_clusters> <initial_records> --flag ...")
        sys.exit(1)
    
    initial_records = int(sys.argv[3])

    # Default
    operation_mode = 'insert_count'
    num_inserted = 0
    insert_start_idx = None
    insert_end_idx = None
    delete_start_idx = None
    delete_end_idx = None
    upsert_start_idx = None
    upsert_end_idx = None

    # New flag forms: --delete-range and --upsert-range (exact placement expected after initial_records)
    if len(sys.argv) == 7 and sys.argv[4] == '--delete-range':
        operation_mode = 'delete_range'
        try:
            delete_start_idx = int(sys.argv[5])
            delete_end_idx = int(sys.argv[6])
        except ValueError:
            print('Error: delete range values must be integers')
            sys.exit(1)
    elif len(sys.argv) == 7 and sys.argv[4] == '--upsert-range':
        operation_mode = 'upsert_range'
        try:
            upsert_start_idx = int(sys.argv[5])
            upsert_end_idx = int(sys.argv[6])
        except ValueError:
            print('Error: upsert range values must be integers')
            sys.exit(1)
    elif len(sys.argv) == 5:
        # 4-arg forms: either insert_count or legacy delete (negative arg)
        arg4 = sys.argv[4]
        if arg4.startswith('-') and arg4[1:].isdigit():
            operation_mode = 'delete'
            delete_start_idx = 0
            delete_end_idx = int(arg4[1:])
        else:
            # insert_count
            try:
                num_inserted = int(arg4)
                operation_mode = 'insert_count'
            except ValueError:
                print('Error: fourth argument must be an integer (num_inserted) or -<end_idx> for delete')
                sys.exit(1)
    else:
        # len == 6 -> insert_range
        operation_mode = 'insert_range'
        try:
            insert_start_idx = int(sys.argv[4])
            insert_end_idx = int(sys.argv[5])
            num_inserted = insert_end_idx - insert_start_idx + 1
        except ValueError:
            print('Error: start/end indices must be integers')
            sys.exit(1)

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(scripts_dir)
    datasets_dir = os.path.join(base_dir, 'datasets')

    initial_file = os.path.join(datasets_dir, f"{dataset_name}_train_{initial_records}.jsonl")
    full_train = os.path.join(datasets_dir, f"{dataset_name}_train.jsonl")

    if not os.path.exists(initial_file):
        print(f"Error: initial subdataset not found: {initial_file}")
        sys.exit(1)

    ds_astx = f"{dataset_name}_{initial_records}".replace('-', '_')

    # Handle delete mode (legacy: loads dataset and creates index before deleting)
    if operation_mode == 'delete':
        # Step 1: Load the initial subdataset into AsterixDB
        print(f'[step] Loading initial subdataset (0 to {initial_records-1}) into AsterixDB')
        run_subprocess([
            sys.executable,
            os.path.join(scripts_dir, 'load_dataset.py'),
            dataset_name,
            str(initial_records)
        ], cwd=base_dir)

        # Step 2: Create vector index with specified number of clusters
        print('[step] Creating vector index')
        run_subprocess([
            sys.executable,
            os.path.join(scripts_dir, 'create_index.py'),
            dataset_name,
            str(num_clusters),
            str(initial_records)
        ], cwd=base_dir)

        # Step 3: Delete records in the specified range
        batch_delete_records(ds_astx, delete_start_idx, delete_end_idx)

        print(f'[done] Loaded {initial_records} records, created index, and deleted records with idx [{delete_start_idx}, {delete_end_idx}]')
        sys.exit(0)

    # Handle delete_range mode (new: skips load/index to preserve existing dataset and index)
    if operation_mode == 'delete_range':
        print('[info] Delete range mode: skipping dataset load and index creation to avoid dropping/recreating index.')
        if not dataset_exists_in_asterix(ds_astx):
            print(f"Error: target AsterixDB dataset '{ds_astx}' does not appear to exist or is not reachable.\nLoad the dataset first before running delete range mode.")
            sys.exit(1)
        
        # Delete records in the specified range
        batch_delete_records(ds_astx, delete_start_idx, delete_end_idx)
        
        print(f'[done] Deleted records with idx [{delete_start_idx}, {delete_end_idx}] from existing dataset')
        sys.exit(0)

    # Handle upsert_range mode (skips load/index to preserve existing dataset and index)
    if operation_mode == 'upsert_range':
        print('[info] Upsert range mode: skipping dataset load and index creation to avoid dropping/recreating index.')
        if not dataset_exists_in_asterix(ds_astx):
            print(f"Error: target AsterixDB dataset '{ds_astx}' does not appear to exist or is not reachable.\nLoad the dataset first before running upsert range mode.")
            sys.exit(1)
        
        # Create upsert file containing records in the specified range
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        num_records = upsert_end_idx - upsert_start_idx + 1
        upsert_filename = f"{dataset_name}_upsert_{num_records}_{timestamp}.jsonl"
        upsert_path = os.path.join(datasets_dir, upsert_filename)
        
        print(f'[step] Preparing upsert file: {upsert_path}')
        written = create_insert_file_by_index(full_train, idx_start=upsert_start_idx, idx_end=upsert_end_idx, out_path=upsert_path)
        
        if written == 0:
            print('[warn] No records were written to upsert file (not enough records in full train or range empty).')
            sys.exit(1)
        
        print(f'[info] Wrote {written} records to {upsert_path}')
        
        # Upsert records into existing AsterixDB dataset
        print(f'[step] Upserting {written} records into AsterixDB dataset: {ds_astx}')
        load_upsert_file_into_asterix(ds_astx, upsert_path)
        
        print(f'[done] Upserted records with idx [{upsert_start_idx}, {upsert_end_idx}]')
        print(f'[info] Upsert file retained at: {upsert_path}')
        sys.exit(0)

    # Handle insert modes (insert_range or insert_count)
    # If running in insert_range mode, avoid reloading dataset and dropping/recreating the index
    if operation_mode == 'insert_range':
        print('[info] Insert range mode: skipping dataset load and pre-insert index creation to avoid dropping/recreating index.')
        if not dataset_exists_in_asterix(ds_astx):
            print(f"Error: target AsterixDB dataset '{ds_astx}' does not appear to exist or is not reachable.\nEither load the initial dataset first or run in insert_count mode to create it.")
            sys.exit(1)
    else:
        # insert_count mode: load dataset and create index
        # Step 1: Load the initial subdataset into AsterixDB
        print('[step] Loading initial subdataset into AsterixDB')
        run_subprocess([
            sys.executable,
            os.path.join(scripts_dir, 'load_dataset.py'),
            dataset_name,
            str(initial_records)
        ], cwd=base_dir)

        # Step 2: Create vector index with specified number of clusters
        print('[step] Creating vector index')
        run_subprocess([
            sys.executable,
            os.path.join(scripts_dir, 'create_index.py'),
            dataset_name,
            str(num_clusters),
            str(initial_records)
        ], cwd=base_dir)

    if num_inserted <= 0:
        print('[info] No additional records to insert. Done.')
        sys.exit(0)

    # Step 3: Create insert file containing next num_inserted records from full train
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    insert_filename = f"{dataset_name}_insert_{num_inserted}_{timestamp}.jsonl"
    insert_path = os.path.join(datasets_dir, insert_filename)

    print(f'[step] Preparing insert file: {insert_path}')
    if operation_mode == 'insert_count':
        written = create_insert_file(full_train, start=initial_records, count=num_inserted, out_path=insert_path)
    else:
        # insert_range: create by idx range
        written = create_insert_file_by_index(full_train, idx_start=insert_start_idx, idx_end=insert_end_idx, out_path=insert_path)

    if written == 0:
        print('[warn] No records were written to insert file (not enough records in full train or range empty).')
        sys.exit(1)

    print(f'[info] Wrote {written} records to {insert_path}')

    # Step 4: Load insert file into existing AsterixDB dataset (append)
    print(f'[step] Inserting {written} records into AsterixDB dataset: {ds_astx}')
    load_insert_file_into_asterix(ds_astx, insert_path)

    print('[done] Dataset extended successfully')
    print(f'[info] Insert file retained at: {insert_path}')


if __name__ == '__main__':
    main()
