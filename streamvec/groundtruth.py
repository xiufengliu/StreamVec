"""Exact ground-truth kNN over the LIVE vector set + recall metrics."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def compute_groundtruth(
    live_vectors: np.ndarray,
    live_ids: Sequence[int],
    queries: np.ndarray,
    k: int = 10,
) -> List[List[int]]:
    """Exact k-NN ground-truth using FAISS IndexFlatL2 over live vectors only.

    Returns a list (per query) of the true neighbor *external ids*.
    """
    import faiss

    if len(live_ids) == 0:
        return [[] for _ in range(len(queries))]
    d = live_vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(live_vectors, dtype=np.float32))
    kk = min(k, len(live_ids))
    _, idx = index.search(np.ascontiguousarray(queries, dtype=np.float32), kk)
    id_arr = np.asarray(live_ids)
    return [id_arr[row].tolist() for row in idx]


def recall_at_k(
    predicted_ids: Sequence[Sequence[int]],
    groundtruth_ids: Sequence[Sequence[int]],
    k: int = 10,
) -> float:
    """Mean Recall@k = average fraction of true top-k retrieved."""
    if not groundtruth_ids:
        return 0.0
    total_hits = 0
    total = 0
    for p, g in zip(predicted_ids, groundtruth_ids):
        gset = set(list(g)[:k])
        if not gset:
            continue
        total_hits += len(set(list(p)[:k]) & gset)
        total += len(gset)
    return total_hits / max(total, 1)


def ndcg_at_k(
    predicted_ids: Sequence[Sequence[int]],
    groundtruth_ids: Sequence[Sequence[int]],
    k: int = 10,
) -> float:
    """NDCG@k with binary relevance (1 if in true top-k)."""
    if not groundtruth_ids:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for p, g in zip(predicted_ids, groundtruth_ids):
        gset = set(list(g)[:k])
        if not gset:
            continue
        rel = np.array([1.0 if pid in gset else 0.0 for pid in list(p)[:k]])
        if len(rel) < k:
            rel = np.pad(rel, (0, k - len(rel)))
        dcg = float(np.sum(rel * discounts))
        ideal = float(np.sum(discounts[: min(len(gset), k)]))
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0
