"""HNSW-style layered proximity graph (base structure for StreamVec).

This is a faithful, self-contained NumPy/Python implementation of the
Hierarchical Navigable Small World graph (Malkov & Yashunin, TPAMI 2020),
exposing the hooks StreamVec needs: tombstone-aware search, per-node live
neighbor counting, heuristic neighbor selection, and edge edit primitives used
by the repair engine.

Design notes
------------
* Vectors live in a contiguous, capacity-doubling float32 matrix; ``id_to_row``
  maps external ids to rows.  Tombstoned vectors keep their row (lazy deletion).
* Query traversal *routes through* tombstoned nodes (so graph connectivity is
  preserved) but never returns them in the result set -- this realizes the
  blueprint's "skip tombstoned nodes ... filter tombstoned nodes from result
  set" while keeping the live subgraph reachable.  StreamVec's repair restores
  live-to-live edges so the live subgraph stays navigable without leaning on an
  ever-growing tombstone mass.
"""
from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .distance import DistanceMetric, distance_batch

IdT = int


class HNSWGraph:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        metric: DistanceMetric = DistanceMetric.L2,
        seed: int = 42,
        initial_capacity: int = 1024,
    ) -> None:
        self.dim = dim
        self.M = M
        self.M0 = 2 * M  # max degree at layer 0 (HNSW convention)
        self.ef_construction = ef_construction
        self.metric = metric
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0
        self._rng = np.random.default_rng(seed)

        self._vectors = np.zeros((initial_capacity, dim), dtype=np.float32)
        self._n_rows = 0
        self.id_to_row: Dict[IdT, int] = {}
        self.row_to_id: List[IdT] = []

        # neighbors[id] = list over layers; each layer is a list of neighbor ids
        self.neighbors: Dict[IdT, List[List[IdT]]] = {}
        self.level: Dict[IdT, int] = {}  # top layer index of each node
        self.entry_point: Optional[IdT] = None
        self.max_level: int = -1

        # bookkeeping for efficiency analysis (implementation-agnostic counts)
        self.dist_calls: int = 0      # vector distance evaluations
        self.neighbor_checks: int = 0  # adjacency entries scanned (incl. dead/tombstoned)
        self.heap_ops: int = 0         # candidate/result heap pushes+pops

    # ------------------------------------------------------------------ #
    # storage helpers
    # ------------------------------------------------------------------ #
    def __contains__(self, node_id: IdT) -> bool:
        return node_id in self.id_to_row

    @property
    def n_nodes(self) -> int:
        return len(self.id_to_row)

    def vector(self, node_id: IdT) -> np.ndarray:
        return self._vectors[self.id_to_row[node_id]]

    def _store_vector(self, node_id: IdT, vec: np.ndarray) -> None:
        if self._n_rows >= self._vectors.shape[0]:
            new_cap = self._vectors.shape[0] * 2
            grown = np.zeros((new_cap, self.dim), dtype=np.float32)
            grown[: self._n_rows] = self._vectors[: self._n_rows]
            self._vectors = grown
        row = self._n_rows
        self._vectors[row] = vec
        self.id_to_row[node_id] = row
        self.row_to_id.append(node_id)
        self._n_rows += 1

    def random_level(self) -> int:
        return int(-math.log(max(self._rng.random(), 1e-12)) * self.mL)

    # ------------------------------------------------------------------ #
    # distance
    # ------------------------------------------------------------------ #
    def _dist_to_many(self, query: np.ndarray, ids: List[IdT]) -> np.ndarray:
        if not ids:
            return np.empty((0,), dtype=np.float32)
        rows = [self.id_to_row[i] for i in ids]
        self.dist_calls += len(ids)
        return distance_batch(query, self._vectors[rows], self.metric)

    # ------------------------------------------------------------------ #
    # core search
    # ------------------------------------------------------------------ #
    def search_layer(
        self,
        query: np.ndarray,
        entry_points: List[IdT],
        ef: int,
        layer: int,
        is_tombstoned: Optional[Callable[[IdT], bool]] = None,
        collect_live_only: bool = False,
        skip_tombstones: bool = False,
    ) -> List[Tuple[float, IdT]]:
        """Greedy beam search within a single layer.

        Returns up to ``ef`` (distance, id) pairs.

        ``collect_live_only`` excludes tombstoned nodes from the returned result
        set.  ``skip_tombstones`` (the blueprint's QUERY semantics) additionally
        refuses to *traverse through* tombstoned nodes: their neighbors are never
        expanded, so a tombstoned node cannot serve as a routing waypoint.  Under
        skipping, deletions fragment the live subgraph unless repair restores
        live-to-live edges -- which is precisely what makes StreamVec's repair
        necessary.
        """
        if is_tombstoned is None:
            is_tombstoned = lambda _i: False  # noqa: E731

        visited = set(entry_points)
        dists = self._dist_to_many(query, entry_points)

        candidates: List[Tuple[float, IdT]] = []  # min-heap on distance
        results: List[Tuple[float, IdT]] = []  # max-heap via negation
        for d, ep in zip(dists, entry_points):
            heapq.heappush(candidates, (d, ep))
            if not ((collect_live_only or skip_tombstones) and is_tombstoned(ep)):
                heapq.heappush(results, (-d, ep))

        while candidates:
            cur_d, cur = heapq.heappop(candidates)
            if results:
                worst = -results[0][0]
                if cur_d > worst and len(results) >= ef:
                    break
            # under skip_tombstones, do not expand the neighbors of a tombstone
            if skip_tombstones and cur not in entry_points and is_tombstoned(cur):
                continue
            nbrs = self.neighbors.get(cur)
            if nbrs is None or layer >= len(nbrs):
                continue
            # the full adjacency list is scanned (incl.\ dead/tombstoned slots):
            # this is the constant-factor scan overhead of skipping.
            self.neighbor_checks += len(nbrs[layer])
            unvisited = [n for n in nbrs[layer] if n not in visited and n in self.id_to_row]
            if skip_tombstones:
                unvisited = [n for n in unvisited if not is_tombstoned(n)]
            if not unvisited:
                continue
            visited.update(unvisited)
            nd = self._dist_to_many(query, unvisited)
            for d, n in zip(nd, unvisited):
                admit_to_results = not (collect_live_only and is_tombstoned(n))
                if len(results) < ef:
                    heapq.heappush(candidates, (d, n))
                    self.heap_ops += 1
                    if admit_to_results:
                        heapq.heappush(results, (-d, n))
                        self.heap_ops += 1
                elif results and d < -results[0][0]:
                    heapq.heappush(candidates, (d, n))
                    self.heap_ops += 1
                    if admit_to_results:
                        heapq.heappush(results, (-d, n))
                        self.heap_ops += 1
                        if len(results) > ef:
                            heapq.heappop(results)
                            self.heap_ops += 1
        out = [(-nd, nid) for nd, nid in results]
        out.sort()
        return out

    # ------------------------------------------------------------------ #
    # heuristic neighbor selection (HNSW Algorithm 4)
    # ------------------------------------------------------------------ #
    def select_neighbors_heuristic(
        self,
        base_id: Optional[IdT],
        candidates: List[Tuple[float, IdT]],
        M: int,
        keep_pruned: bool = True,
    ) -> List[IdT]:
        """Diversifying neighbor selection (HNSW Alg. 4), vectorized.

        Keep a candidate only if it is closer to the base node than to every
        already-selected neighbor.  Distances from a candidate to the current
        selected set are computed in a single batch call for speed.
        """
        if len(candidates) <= M:
            return [c for _, c in sorted(candidates)]
        cand = sorted(candidates)  # ascending distance to base
        cand_ids = [c for _, c in cand]
        cand_d = np.asarray([d for d, _ in cand], dtype=np.float32)
        cand_rows = [self.id_to_row[c] for c in cand_ids]
        cand_vecs = self._vectors[cand_rows]
        C = len(cand_ids)

        # full candidate-candidate distance matrix in ONE batched computation
        # (replaces O(C^2) tiny per-pair calls)
        pdm = self._pairwise_matrix(cand_vecs)
        self.dist_calls += C * (C - 1) // 2

        sel_idx: List[int] = []
        pruned: List[IdT] = []
        for i in range(C):
            if len(sel_idx) >= M:
                break
            if not sel_idx:
                sel_idx.append(i)
                continue
            # min distance from candidate i to any already-selected candidate
            if float(pdm[i, sel_idx].min()) >= cand_d[i]:
                sel_idx.append(i)
            else:
                pruned.append(cand_ids[i])
        selected = [cand_ids[i] for i in sel_idx]
        if keep_pruned:
            for c in pruned:
                if len(selected) >= M:
                    break
                selected.append(c)
        return selected

    def _pairwise_matrix(self, vecs: np.ndarray) -> np.ndarray:
        """Pairwise distance matrix among rows of ``vecs`` for the active metric.

        L2 / cosine -> squared Euclidean (cosine vecs are unit-norm, so squared
        Euclidean is monotone in cosine).  IP -> negated inner product.
        """
        if self.metric == DistanceMetric.IP:
            return (-(vecs @ vecs.T)).astype(np.float32)
        sq = np.einsum("ij,ij->i", vecs, vecs)
        gram = vecs @ vecs.T
        dm = sq[:, None] + sq[None, :] - 2.0 * gram
        np.maximum(dm, 0.0, out=dm)
        return dm.astype(np.float32)

    # ------------------------------------------------------------------ #
    # mutation primitives
    # ------------------------------------------------------------------ #
    def _set_neighbors(self, node_id: IdT, layer: int, nbrs: List[IdT]) -> None:
        self.neighbors[node_id][layer] = list(nbrs)

    def _connect(self, a: IdT, b: IdT, layer: int) -> None:
        na = self.neighbors.get(a)
        if na is None or layer >= len(na):
            return  # a does not exist at this layer; skip invalid edge
        if b not in na[layer]:
            na[layer].append(b)

    def add_node(
        self,
        node_id: IdT,
        vec: np.ndarray,
        is_tombstoned: Optional[Callable[[IdT], bool]] = None,
        level: Optional[int] = None,
    ) -> int:
        """Insert a node into the graph (standard HNSW insertion).

        Returns the node's assigned top level.  ``is_tombstoned`` lets insertion
        route through tombstoned nodes without selecting them as neighbors.
        """
        if is_tombstoned is None:
            is_tombstoned = lambda _i: False  # noqa: E731
        vec = np.ascontiguousarray(vec, dtype=np.float32)
        if node_id not in self.id_to_row:
            self._store_vector(node_id, vec)
        else:
            self._vectors[self.id_to_row[node_id]] = vec

        node_level = self.random_level() if level is None else level
        self.level[node_id] = node_level
        self.neighbors[node_id] = [[] for _ in range(node_level + 1)]

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = node_level
            return node_level

        ep = [self.entry_point]
        # descend from top to node_level+1 with ef=1
        for lc in range(self.max_level, node_level, -1):
            res = self.search_layer(vec, ep, ef=1, layer=lc, is_tombstoned=is_tombstoned)
            if res:
                ep = [res[0][1]]

        # connect at each layer from min(node_level, max_level) down to 0
        for lc in range(min(node_level, self.max_level), -1, -1):
            cand = self.search_layer(vec, ep, ef=self.ef_construction, layer=lc, is_tombstoned=is_tombstoned)
            # prefer live candidates as neighbors
            live_cand = [(d, c) for d, c in cand if not is_tombstoned(c)]
            use_cand = live_cand if live_cand else cand
            m = self.M0 if lc == 0 else self.M
            chosen = self.select_neighbors_heuristic(node_id, use_cand, m)
            # only keep neighbors that actually exist at layer lc (HNSW invariant)
            chosen = [c for c in chosen if self.level.get(c, 0) >= lc]
            self._set_neighbors(node_id, lc, chosen)
            for nb in chosen:
                self._connect(nb, node_id, lc)
                self._prune_if_needed(nb, lc)
            if cand:
                ep = [c for _, c in cand]

        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = node_id
        return node_level

    def _prune_if_needed(self, node_id: IdT, layer: int) -> None:
        """Prune an over-full neighbor list back to ``m_max`` using the
        diversifying heuristic (needed for graph quality on clustered data).

        A small slack lets the list grow a few edges past ``m_max`` before we
        pay for a prune, so the O(deg^2) heuristic fires once every few edge
        insertions instead of on every reverse edge -- a big constant-factor win
        with no quality loss (we still prune *to* m_max with the full heuristic).
        """
        m_max = self.M0 if layer == 0 else self.M
        nbrs = self.neighbors[node_id][layer]
        slack = 1 + m_max // 4
        if len(nbrs) <= m_max + slack:
            return
        base = self.vector(node_id)
        dists = self._dist_to_many(base, nbrs)
        cand = list(zip(dists.tolist(), nbrs))
        chosen = self.select_neighbors_heuristic(node_id, cand, m_max)
        self._set_neighbors(node_id, layer, chosen)

    def live_neighbor_count(self, node_id: IdT, layer: int, is_tombstoned: Callable[[IdT], bool]) -> int:
        nbrs = self.neighbors.get(node_id)
        if nbrs is None or layer >= len(nbrs):
            return 0
        return sum(1 for n in nbrs[layer] if n in self.id_to_row and not is_tombstoned(n))
