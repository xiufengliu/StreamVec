"""Dataset loaders: real SIFT (parquet) and controllable synthetic data."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_RAW = os.path.join(os.path.dirname(_HERE), "data", "raw")


_DATA_PROC = os.path.join(os.path.dirname(_HERE), "data", "processed")


def load_sift(
    n_base: Optional[int] = None,
    n_query: Optional[int] = None,
    data_dir: str = _DATA_RAW,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the SIFT-128 dataset.

    Prefers the small cached ``data/processed/*.npy`` (written by
    ``experiments/prepare_data.py``) to keep memory low when many experiment
    processes run concurrently; falls back to the full parquet otherwise.
    Subsamples deterministically when ``n_base`` / ``n_query`` are given.
    """
    cache_base = os.path.join(_DATA_PROC, "sift_base.npy")
    cache_query = os.path.join(_DATA_PROC, "sift_query.npy")
    if os.path.exists(cache_base) and os.path.exists(cache_query):
        base = np.load(cache_base, mmap_mode="r")
        queries = np.load(cache_query, mmap_mode="r")
        base = np.ascontiguousarray(base)
        queries = np.ascontiguousarray(queries)
    else:
        import pyarrow.parquet as pq
        train = pq.read_table(os.path.join(data_dir, "sift_train.parquet"))
        base = np.vstack(train.column("emb").to_numpy()).astype(np.float32)
        test = pq.read_table(os.path.join(data_dir, "sift_test.parquet"))
        queries = np.vstack(test.column("emb").to_numpy()).astype(np.float32)

    rng = np.random.default_rng(seed)
    if n_base is not None and n_base < len(base):
        idx = rng.choice(len(base), size=n_base, replace=False)
        base = base[np.sort(idx)]
    if n_query is not None and n_query < len(queries):
        idx = rng.choice(len(queries), size=n_query, replace=False)
        queries = queries[np.sort(idx)]
    return np.ascontiguousarray(base), np.ascontiguousarray(queries)


def make_synthetic(
    n_base: int,
    dim: int = 128,
    n_query: int = 1000,
    n_clusters: int = 50,
    cluster_std: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-mixture synthetic vectors (controllable N and D for scaling)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, size=(n_clusters, dim)).astype(np.float32)
    base_assign = rng.integers(0, n_clusters, size=n_base)
    base = (centers[base_assign] + rng.normal(0, cluster_std, size=(n_base, dim))).astype(np.float32)
    q_assign = rng.integers(0, n_clusters, size=n_query)
    queries = (centers[q_assign] + rng.normal(0, cluster_std, size=(n_query, dim))).astype(np.float32)
    return np.ascontiguousarray(base), np.ascontiguousarray(queries)


def load_cached_npy(prefix: str, n_base=None, n_query=None, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    """Load a cached (base, query) pair from data/processed/<prefix>_{base,query}.npy."""
    base = np.ascontiguousarray(np.load(os.path.join(_DATA_PROC, f"{prefix}_base.npy"), mmap_mode="r"))
    queries = np.ascontiguousarray(np.load(os.path.join(_DATA_PROC, f"{prefix}_query.npy"), mmap_mode="r"))
    rng = np.random.default_rng(seed)
    if n_base is not None and n_base < len(base):
        base = base[np.sort(rng.choice(len(base), size=n_base, replace=False))]
    if n_query is not None and n_query < len(queries):
        queries = queries[np.sort(rng.choice(len(queries), size=n_query, replace=False))]
    return np.ascontiguousarray(base), np.ascontiguousarray(queries)


def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    name = name.lower()
    if name == "sift":
        return load_sift(**{k: v for k, v in kwargs.items() if k in ("n_base", "n_query", "seed")})
    if name == "glove":
        return load_cached_npy("glove", **{k: v for k, v in kwargs.items() if k in ("n_base", "n_query", "seed")})
    if name == "synthetic":
        return make_synthetic(**{k: v for k, v in kwargs.items()
                                 if k in ("n_base", "dim", "n_query", "n_clusters", "cluster_std", "seed")})
    raise ValueError(f"unknown dataset {name!r}")
