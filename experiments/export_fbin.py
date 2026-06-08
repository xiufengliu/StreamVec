"""Export SIFT base + queries to flat .fbin (int32 n, int32 dim, then float32 data)
for fast C++ loading.  Streams the parquet so it never loads all 1M into RAM."""
import os, sys, struct
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "bin")
os.makedirs(OUT, exist_ok=True)


def write_fbin(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", arr.shape[0], arr.shape[1]))
        arr.tofile(f)
    print(f"wrote {path}: {arr.shape}")


def main():
    import pyarrow.parquet as pq
    n_base = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    pf = pq.ParquetFile(os.path.join(RAW, "sift_train.parquet"))
    rows, got = [], 0
    for b in pf.iter_batches(batch_size=20000, columns=["emb"]):
        rows.append(np.array(b.column("emb").to_pylist(), dtype=np.float32))
        got += len(rows[-1])
        if got >= n_base:
            break
    base = np.vstack(rows)[:n_base]
    te = pq.read_table(os.path.join(RAW, "sift_test.parquet"))
    q = np.array(te.column("emb").to_pylist(), dtype=np.float32)
    write_fbin(os.path.join(OUT, "sift_base.fbin"), base)
    write_fbin(os.path.join(OUT, "sift_query.fbin"), q)


if __name__ == "__main__":
    main()
