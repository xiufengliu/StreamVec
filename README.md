# StreamVec

**Efficient incremental index maintenance for approximate nearest neighbor (ANN) search over continuously updated vector streams.**

StreamVec is a streaming maintenance layer for HNSW-style proximity graphs that
keeps a high-quality ANN index under a mixed insert/delete/update stream.

## What StreamVec does

1. **Lazy tombstone deletion** — `O(M)` soft delete; no immediate rewiring.
2. **Tombstone-skipping traversal** — queries never route through deleted nodes, so query cost is invariant to the deletion ratio.
3. **Degree-deficit priority repair (`DirtySet`)** — budgeted repair of the most-degraded live nodes.
4. **Adaptive rebuild trigger** — bounds the tombstone ratio and memory.

## Layout

```
streamvec/        core Python library (index, hnsw_graph, dirty_set, repair,
                  tombstone, baselines, datasets, stream_gen, groundtruth, ...)
cpp/              C++ harness (streamvec.hpp, index.hpp, baselines.hpp,
                  bench.cpp) + Makefile
experiments/      run_unit.py (resumable driver), units.py (configs),
                  exp* wrappers, prepare_data.py, make_figures.py
results/          per-experiment CSV/JSON outputs
results_cpp/      C++ harness outputs
figures/          generated PDF/PNG figures
tests/            test_smoke.py (sanity + recall checks)
```

> **Note.** `data/` (raw/processed vectors) is not tracked in git (datasets
> exceed GitHub size limits). Regenerate it with `experiments/prepare_data.py`.

## Quickstart (Python)

```bash
pip install -r requirements.txt
python experiments/prepare_data.py          # cache a dataset subset (needs data/raw/*.parquet)
python -m tests.test_smoke                  # sanity + recall checks
python experiments/run_unit.py all 30       # run pending experiment units (resumable)
python experiments/make_figures.py          # regenerate figures
```

The runner is **resumable**: each invocation runs as many units as fit in the
given wall-clock budget (seconds), appends to `results/`, and exits; re-invoke to
continue. Single experiments: `python experiments/run_unit.py exp3 30`.

## C++ harness

```bash
cd cpp && make            # g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 -pthread
./bench <mode> ...        # modes selected by argv[1]; see bench.cpp
```
