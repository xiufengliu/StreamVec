# StreamVec

Streaming maintenance for HNSW-style approximate nearest neighbor (ANN) graphs
under a mixed stream of inserts, deletes and updates.

StreamVec keeps an in-memory HNSW graph usable while the underlying vector
collection changes:

1. **Lazy tombstone deletion** — a delete flags the node and records which live
   neighbors were affected; no edges are rewired on the delete path.
2. **Tombstone-skipping search** — queries never evaluate or return deleted nodes,
   so a search on the graph behaves like a search on its live subgraph.
3. **Degree-deficit DirtySet** — damaged live nodes are kept in a max-heap keyed by
   their number of missing neighbors; a per-interval budget of them is repaired by
   local beam search, the HNSW neighbor heuristic and reverse edges.
4. **Adaptive trigger** — when the tombstone ratio exceeds a threshold, the index
   is rebuilt or compacted, bounding tombstone memory.

The repository also contains the alternative maintenance strategies used for
comparison, all built on the same graph code: periodic rebuild, route-through
tombstones (`mark_deleted`-style append), slot reuse (`replace_deleted`-style),
a base + delta overlay, and FreshDiskANN-style delete consolidation with
compaction.

## Layout

```
cpp/              C++17 implementation and benchmark harness
  streamvec.hpp   HNSW graph, SIMD distance, DirtySet
  index.hpp       StreamVecIndex: insert/delete/update, repair, trigger, rebuild
  baselines.hpp   periodic rebuild, delta overlay, delete consolidation
  bench.cpp       experiment driver (modes below)
streamvec/        pure-Python/NumPy reference implementation
experiments/      Python experiment units, data preparation, figure and
                  aggregation scripts
tests/            Python smoke tests; tests/cpp/ trace-equivalence check
results/          Python experiment outputs (CSV/JSON)
results_cpp/      C++ experiment outputs (CSV); v2/v3/v4 hold the multi-seed runs
figures/          generated figures (PDF/PNG)
```

## Data

`data/` is not tracked. Prepare SIFT/GloVe subsets with

```bash
pip install -r requirements.txt
python experiments/prepare_data.py     # caches subsets as .npy (needs data/raw/*.parquet)
python experiments/export_fbin.py      # writes data/bin/*.fbin for the C++ harness
```

The C++ harness reads `data/bin/sift_base.fbin` and `data/bin/sift_query.fbin` by
default (`int32 n, int32 d, float32[n*d]`); set `SV_BASE` / `SV_QUERY` to use
another file pair (for cosine, normalize the vectors first).

## C++ harness

```bash
cd cpp && make        # g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 -pthread
cd ..                 # run from the repository root (data paths are relative)
./cpp/bench <mode> [args]
```

| Mode | Arguments | Output |
|------|-----------|--------|
| `e2e2` | `<system> <N> <ops> <seed>` | time-averaged recall during a mixed stream, final recall/QPS at ef 10/40/100, update throughput, memory → `e2e_v2.csv` |
| `e2e3` | `<system> <N> <ops> <seed>` | as `e2e2` plus time-averaged single-thread QPS and repair diagnostics → `e2e3.csv` |
| `fresh2` | `<system> <R> <seed>` | Recall@10 against the live set every 2k operations → `fresh_v2.csv` |
| `churn` | `<system> <N>` | recall and QPS while deleting up to 40% → `churn.csv` |
| `scale` | `<N>` | build/insert/query latency and memory → `scale.csv` |
| `concur` | `<threads> <N>` | query QPS and tail latency with a concurrent delete+repair writer → `concur2.csv` |

Systems: `streamvec`, `svc` (StreamVec with compaction on trigger), `svc0`
(skipping + compaction, no repair), `consolidate`, `append`, `reuse`, `rebuild`,
`delta`, `norepair`.

Environment overrides: `SV_RES` (output prefix, default `results_cpp/`),
`SV_M`, `SV_EFC`, `SV_B` (repair budget), `SV_IV` (repair interval), `SV_TH`
(tombstone threshold), `SV_POL` (0 priority, 1 FIFO, 2 random, 3 eager),
`SV_P` (dead-fraction threshold for immediate repair), `SV_DEFTH` (deficit
trigger), `SV_R` (rebuild/merge interval), `SV_MIX` (`i,d,u` operation mix),
`SV_NOWRITER` (concurrency baseline without writer).

Example (three seeds, M = 16):

```bash
for s in 42 7 123; do
  for sys in streamvec consolidate svc0; do
    SV_RES=results_cpp/v3/M16_ SV_M=16 ./cpp/bench e2e2 $sys 60000 60000 $s
  done
done
python experiments/aggregate_cpp_v2.py   # summary.json, tables/*.tex, figures/cpp_*_v2.pdf
python experiments/aggregate_cpp_v4.py   # dataset/workload generality summary
```

## Python reference implementation

```bash
python -m tests.test_smoke                  # sanity + recall checks
python experiments/run_unit.py all 30       # resumable experiment units
python experiments/make_figures.py          # figures from results/
```

## Trace-equivalence check

`tests/cpp/trace_equivalence_check.cpp` verifies that tombstone-skipping search on
a graph with tombstones performs exactly the same distance evaluations, heap
operations and result list as an unmodified search on the graph with tombstoned
entries removed.

```bash
g++ -O2 -std=c++17 -fopenmp tests/cpp/trace_equivalence_check.cpp -o trace_check
./trace_check > tests/cpp/trace_equivalence_result.json
```
