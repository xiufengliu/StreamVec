"""Latency / throughput / memory measurement helpers."""
from __future__ import annotations

import time
from typing import List

import numpy as np


class Timer:
    def __init__(self) -> None:
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0


def percentiles(latencies_ms: List[float]) -> dict:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def estimate_index_memory_mb(index) -> float:
    """Rough resident memory of a StreamVecIndex: vectors + adjacency lists."""
    g = index.graph
    vec_bytes = g._n_rows * g.dim * 4
    edge_count = 0
    for nbrs in g.neighbors.values():
        for layer in nbrs:
            edge_count += len(layer)
    # 8 bytes per neighbor id (python int in list ~ pointer + small int cache,
    # but we report the structural cost: id storage)
    edge_bytes = edge_count * 8
    orig_bytes = len(index._orig_vectors) * g.dim * 4
    return (vec_bytes + edge_bytes + orig_bytes) / (1024 * 1024)
