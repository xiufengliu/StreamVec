"""One-time data prep: cache a subsampled SIFT base + query set as .npy so that
each experiment process loads a small file (instead of the full 512MB parquet,
which OOMs when several experiments run concurrently)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
PROC = os.path.join(HERE, "data", "processed")
N_BASE = 50000   # superset; experiments subsample from this deterministically
N_QUERY = 1000


def main():
    import pyarrow.parquet as pq
    os.makedirs(PROC, exist_ok=True)
    # read only the first N_BASE rows via row-group-limited iteration (low memory)
    pf = pq.ParquetFile(os.path.join(RAW, "sift_train.parquet"))
    rows = []
    got = 0
    for batch in pf.iter_batches(batch_size=10000, columns=["emb"]):
        arr = np.array(batch.column("emb").to_pylist(), dtype=np.float32)
        rows.append(arr)
        got += len(arr)
        if got >= N_BASE:
            break
    base = np.vstack(rows)[:N_BASE]
    test = pq.read_table(os.path.join(RAW, "sift_test.parquet"))
    queries = np.array(test.column("emb").to_pylist(), dtype=np.float32)[:N_QUERY]
    np.save(os.path.join(PROC, "sift_base.npy"), base)
    np.save(os.path.join(PROC, "sift_query.npy"), queries)
    print(f"cached SIFT base={base.shape} query={queries.shape} -> {PROC}")


if __name__ == "__main__":
    main()
