"""Distance computation for StreamVec.

Vectorized NumPy implementations of L2, cosine, and inner-product distances.
The blueprint specifies SIMD/AVX2 distance kernels in C++; in this Python
reference we rely on NumPy's BLAS-backed batch operations, which is the
idiomatic vectorized equivalent. All distances are *smaller-is-closer* so the
same greedy-search and neighbor-selection logic works for every metric.
"""
from __future__ import annotations

import enum
import numpy as np


class DistanceMetric(enum.Enum):
    L2 = "l2"
    COSINE = "cosine"
    IP = "ip"  # inner product (negated so smaller = closer)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows; used for cosine so cosine distance reduces to L2."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype(np.float32)


def preprocess(vectors: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Apply metric-specific preprocessing (normalization for cosine)."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if metric == DistanceMetric.COSINE:
        return normalize(vectors)
    return vectors


def distance(a: np.ndarray, b: np.ndarray, metric: DistanceMetric) -> float:
    """Distance between two 1-D vectors (already preprocessed for cosine)."""
    if metric == DistanceMetric.L2:
        d = a - b
        return float(np.dot(d, d))
    if metric == DistanceMetric.COSINE:
        # vectors are unit-normalized => squared-L2 == 2 - 2*cos
        d = a - b
        return float(np.dot(d, d))
    # inner product, negated so that smaller is closer
    return float(-np.dot(a, b))


def distance_batch(query: np.ndarray, mat: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Distance from one query (1-D) to many vectors (2-D), BLAS-accelerated.

    Returns a 1-D array of distances (smaller = closer).
    """
    if mat.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if metric == DistanceMetric.L2:
        # ||q-x||^2 = ||q||^2 + ||x||^2 - 2 q.x  (kept numerically simple)
        diff = mat - query
        return np.einsum("ij,ij->i", diff, diff).astype(np.float32)
    if metric == DistanceMetric.COSINE:
        diff = mat - query  # unit vectors -> squared L2 monotone in cosine
        return np.einsum("ij,ij->i", diff, diff).astype(np.float32)
    return (-(mat @ query)).astype(np.float32)
