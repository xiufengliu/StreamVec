"""Baseline streaming-index strategies, with a uniform interface.

Every baseline exposes the same streaming API as StreamVecIndex:
``insert(id, vec)``, ``remove(id)``, ``update(id, vec)``, ``search(q, k, ef)``,
``maybe_maintain(op_count)`` (called once per stream op so periodic-rebuild
baselines can rebuild on schedule), and ``live_vectors()``.

Baselines (from blueprint Part 4.3 and the related-work survey):
  * HNSWAppend       -- hnswlib insert + mark_deleted, never repaired/rebuilt.
  * HNSWRebuild      -- buffers ops, full hnswlib rebuild every K ops.
  * FAISSIVFRebuild  -- FAISS IVF, full rebuild every K ops.
  * StreamVecEager   -- StreamVec with eager (repair-all-immediately) policy;
                        implemented in experiments via repair_policy='eager'.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
class PyRebuild:
    """Periodic full rebuild using StreamVec's OWN Python HNSW graph.

    This is the implementation-matched counterpart to the StreamVec index: same
    language, same graph code, same search.  It buffers updates and rebuilds the
    whole graph from the live set every ``rebuild_interval`` operations.  It
    isolates the *algorithmic* effect (incremental repair vs. periodic rebuild)
    from the C++-vs-Python language gap, so the update-throughput comparison is
    fair.
    """

    name = "Python-Rebuild"

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 rebuild_interval: int = 4000, metric: str = "l2", **_):
        from .hnsw_graph import HNSWGraph
        from .distance import DistanceMetric
        self._HNSWGraph = HNSWGraph
        self._metric = {"l2": DistanceMetric.L2, "cosine": DistanceMetric.COSINE,
                        "ip": DistanceMetric.IP}[metric]
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.rebuild_interval = rebuild_interval
        self._live: Dict[int, np.ndarray] = {}
        self._since = 0
        self.rebuild_count = 0
        self._graph = None
        self._build()

    def _build(self):
        from .distance import preprocess
        g = self._HNSWGraph(self.dim, M=self.M, ef_construction=self.ef_construction,
                            metric=self._metric, seed=42)
        for nid, vec in self._live.items():
            g.add_node(nid, preprocess(vec, self._metric))
        self._graph = g
        self._since = 0
        self.rebuild_count += 1

    def insert(self, node_id: int, vec: np.ndarray) -> None:
        self._live[node_id] = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._since += 1

    def remove(self, node_id: int) -> None:
        self._live.pop(node_id, None)
        self._since += 1

    def update(self, node_id: int, vec: np.ndarray) -> None:
        self._live[node_id] = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._since += 1

    def maybe_maintain(self, op_count: int) -> None:
        if self._since >= self.rebuild_interval:
            self._build()

    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 50):
        from .distance import preprocess
        if self._graph is None or self._graph.entry_point is None or not self._live:
            return []
        q = preprocess(np.asarray(query, dtype=np.float32).reshape(-1), self._metric)
        g = self._graph
        ep = [g.entry_point]
        for lc in range(g.max_level, 0, -1):
            res = g.search_layer(q, ep, ef=1, layer=lc)
            if res:
                ep = [res[0][1]]
        res = g.search_layer(q, ep, ef=max(ef_search, k), layer=0)
        return [(d, i) for d, i in res[:k]]

    def live_vectors(self):
        if not self._live:
            return np.empty((0, self.dim), dtype=np.float32), []
        ids = list(self._live.keys())
        return np.vstack([self._live[i] for i in ids]).astype(np.float32), ids


# --------------------------------------------------------------------------- #
class HNSWAppend:
    """Append-only HNSW (hnswlib) with mark_deleted soft deletion, no repair."""

    name = "HNSW-Append"

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 capacity: int = 2_000_000, metric: str = "l2", **_):
        import hnswlib
        space = {"l2": "l2", "cosine": "cosine", "ip": "ip"}[metric]
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=capacity, ef_construction=ef_construction, M=M)
        self.dim = dim
        self._live: Dict[int, np.ndarray] = {}
        self._deleted: set = set()

    def insert(self, node_id: int, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        self.index.add_items(v, np.array([node_id]))
        self._live[node_id] = v.reshape(-1)
        self._deleted.discard(node_id)

    def remove(self, node_id: int) -> None:
        try:
            self.index.mark_deleted(node_id)
        except RuntimeError:
            pass
        self._live.pop(node_id, None)
        self._deleted.add(node_id)

    def update(self, node_id: int, vec: np.ndarray) -> None:
        self.remove(node_id)
        # reinsert with a fresh id space collision avoided by hnswlib replace
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        try:
            self.index.add_items(v, np.array([node_id]), replace_deleted=True)
        except Exception:
            self.index.add_items(v, np.array([node_id]))
        self._live[node_id] = v.reshape(-1)
        self._deleted.discard(node_id)

    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 50) -> List[Tuple[float, int]]:
        if not self._live:
            return []
        self.index.set_ef(max(ef_search, k))
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        kk = min(k, len(self._live))
        try:
            labels, dists = self.index.knn_query(q, k=kk)
        except RuntimeError:
            return []
        return [(float(d), int(l)) for d, l in zip(dists[0], labels[0])]

    def maybe_maintain(self, op_count: int) -> None:
        return None

    def live_vectors(self):
        if not self._live:
            return np.empty((0, self.dim), dtype=np.float32), []
        ids = list(self._live.keys())
        return np.vstack([self._live[i] for i in ids]).astype(np.float32), ids


# --------------------------------------------------------------------------- #
class HNSWRebuild:
    """HNSW that performs a full rebuild every ``rebuild_interval`` operations."""

    name = "HNSW-Rebuild"

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 rebuild_interval: int = 50_000, metric: str = "l2", **_):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.rebuild_interval = rebuild_interval
        self.metric = metric
        self._live: Dict[int, np.ndarray] = {}
        self._dirty_since_rebuild = 0
        self.rebuild_count = 0
        self._index = None
        self._build()

    def _build(self):
        import hnswlib
        space = {"l2": "l2", "cosine": "cosine", "ip": "ip"}[self.metric]
        idx = hnswlib.Index(space=space, dim=self.dim)
        cap = max(len(self._live) * 2, 1024)
        idx.init_index(max_elements=cap, ef_construction=self.ef_construction, M=self.M)
        if self._live:
            ids = np.array(list(self._live.keys()))
            mat = np.vstack([self._live[i] for i in self._live]).astype(np.float32)
            idx.add_items(mat, ids)
        self._index = idx
        self._dirty_since_rebuild = 0
        self.rebuild_count += 1

    def insert(self, node_id: int, vec: np.ndarray) -> None:
        self._live[node_id] = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._dirty_since_rebuild += 1

    def remove(self, node_id: int) -> None:
        self._live.pop(node_id, None)
        self._dirty_since_rebuild += 1

    def update(self, node_id: int, vec: np.ndarray) -> None:
        self._live[node_id] = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._dirty_since_rebuild += 1

    def maybe_maintain(self, op_count: int) -> None:
        if self._dirty_since_rebuild >= self.rebuild_interval:
            self._build()

    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 50) -> List[Tuple[float, int]]:
        if not self._live or self._index is None:
            return []
        self._index.set_ef(max(ef_search, k))
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        kk = min(k, len(self._live))
        labels, dists = self._index.knn_query(q, k=kk)
        return [(float(d), int(l)) for d, l in zip(dists[0], labels[0])]

    def live_vectors(self):
        if not self._live:
            return np.empty((0, self.dim), dtype=np.float32), []
        ids = list(self._live.keys())
        return np.vstack([self._live[i] for i in ids]).astype(np.float32), ids


# --------------------------------------------------------------------------- #
class FAISSIVFRebuild:
    """FAISS IVF-Flat that fully rebuilds every ``rebuild_interval`` ops."""

    name = "FAISS-IVF-Rebuild"

    def __init__(self, dim: int, nlist: int = 100, nprobe: int = 16,
                 rebuild_interval: int = 50_000, metric: str = "l2", **_):
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.rebuild_interval = rebuild_interval
        self.metric = metric
        self._live: Dict[int, np.ndarray] = {}
        self._dirty_since_rebuild = 0
        self.rebuild_count = 0
        self._index = None
        self._ids: List[int] = []
        self._build()

    def _build(self):
        import faiss
        if len(self._live) < self.nlist:
            self._index = None
            self._dirty_since_rebuild = 0
            self.rebuild_count += 1
            return
        ids = list(self._live.keys())
        mat = np.vstack([self._live[i] for i in ids]).astype(np.float32)
        quant = faiss.IndexFlatL2(self.dim)
        nlist = min(self.nlist, max(1, len(ids) // 39))
        index = faiss.IndexIVFFlat(quant, self.dim, nlist)
        index.train(mat)
        index.add(mat)
        index.nprobe = self.nprobe
        self._index = index
        self._ids = ids
        self._dirty_since_rebuild = 0
        self.rebuild_count += 1

    def insert(self, node_id: int, vec: np.ndarray) -> None:
        self._live[node_id] = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._dirty_since_rebuild += 1

    def remove(self, node_id: int) -> None:
        self._live.pop(node_id, None)
        self._dirty_since_rebuild += 1

    def update(self, node_id: int, vec: np.ndarray) -> None:
        self._live[node_id] = np.asarray(vec, dtype=np.float32).reshape(-1)
        self._dirty_since_rebuild += 1

    def maybe_maintain(self, op_count: int) -> None:
        if self._dirty_since_rebuild >= self.rebuild_interval:
            self._build()

    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 50) -> List[Tuple[float, int]]:
        if self._index is None or not self._live:
            return []
        # ef_search maps to nprobe for IVF (search-effort knob)
        self._index.nprobe = max(1, min(self.nlist, ef_search))
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        kk = min(k, len(self._ids))
        D, I = self._index.search(q, kk)
        return [(float(d), int(self._ids[i])) for d, i in zip(D[0], I[0]) if i >= 0]

    def live_vectors(self):
        if not self._live:
            return np.empty((0, self.dim), dtype=np.float32), []
        ids = list(self._live.keys())
        return np.vstack([self._live[i] for i in ids]).astype(np.float32), ids
