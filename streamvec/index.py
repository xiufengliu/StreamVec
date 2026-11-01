"""StreamVecIndex: the main streaming ANN index.

Implements the core operations -- ``insert``, ``remove`` (lazy
tombstone), ``update`` (delete + reinsert), ``search`` (tombstone-aware),
``repair(budget)`` (priority-guided), ``should_rebuild`` / ``rebuild``
(adaptive trigger), plus ``get_stats`` and ``estimate_recall``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .distance import DistanceMetric, preprocess
from .dirty_set import DirtySet
from .hnsw_graph import HNSWGraph
from .repair import RepairEngine
from .tombstone import TombstoneManager

IdT = int


@dataclass
class IndexStats:
    n_live: int = 0
    n_tombstone: int = 0
    n_dirty: int = 0
    tombstone_ratio: float = 0.0
    avg_degree_deficit: float = 0.0
    insert_count: int = 0
    delete_count: int = 0
    update_count: int = 0
    repair_count: int = 0
    rebuild_count: int = 0
    dist_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class StreamVecIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        metric: DistanceMetric = DistanceMetric.L2,
        tombstone_threshold: float = 0.15,
        degree_deficit_threshold: float = 0.3,
        recall_drop_threshold: float = 0.05,
        tombstone_prune_fraction: float = 0.5,
        repair_policy: str = "priority",
        ef_repair: int = 50,
        hotness_alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.metric = metric
        self.tombstone_threshold = tombstone_threshold
        self.degree_deficit_threshold = degree_deficit_threshold
        self.recall_drop_threshold = recall_drop_threshold
        self.tombstone_prune_fraction = tombstone_prune_fraction
        self.seed = seed

        # 'hotness' is priority ordering with a query-frequency-weighted key
        self._hotness = repair_policy == "hotness"
        self.hotness_alpha = hotness_alpha
        dirty_policy = "priority" if self._hotness else repair_policy

        self.graph = HNSWGraph(dim, M=M, ef_construction=ef_construction, metric=metric, seed=seed)
        self.tomb = TombstoneManager()
        self.dirty = DirtySet(policy=dirty_policy, seed=seed)
        self.repair_engine = RepairEngine(ef_repair=ef_repair)

        # original vectors retained for rebuild and recall estimation
        self._orig_vectors: Dict[IdT, np.ndarray] = {}
        self._version_alias: Dict[IdT, IdT] = {}  # internal version id -> external id
        self._version_counter: int = 0
        self.visit_count: Dict[IdT, int] = {}     # query-access frequency per internal node
        self.stats = IndexStats()
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _target_degree(self, node_id: IdT) -> int:
        # repair targets layer-0 degree (where most edges live)
        return self.graph.M0

    def _mark_neighbors_dirty(self, deleted_id: IdT) -> None:
        """A node was tombstoned: every live neighbor lost a neighbor."""
        nbrs = self.graph.neighbors.get(deleted_id)
        if nbrs is None:
            return
        affected = set()
        for layer_nbrs in nbrs:
            for u in layer_nbrs:
                if u in self.graph.id_to_row and not self.tomb.is_tombstoned(u):
                    affected.add(u)
        for u in affected:
            deficit = self._target_degree(u) - self.graph.live_neighbor_count(u, 0, self.tomb.is_tombstoned)
            if self._hotness:
                # weight repair priority by query-access frequency so hot,
                # damaged nodes are repaired first under skewed workloads
                priority = deficit + self.hotness_alpha * self.visit_count.get(u, 0)
                self.dirty.mark_dirty(u, deficit, priority=priority)
            else:
                self.dirty.mark_dirty(u, deficit)
            # tombstone pruning: force-repair nodes with too many dead neighbors
            self._maybe_force_repair(u)

    def _maybe_force_repair(self, u: IdT) -> None:
        nbrs0 = self.graph.neighbors.get(u)
        if not nbrs0:
            return
        layer0 = nbrs0[0]
        if not layer0:
            return
        dead = sum(1 for n in layer0 if n not in self.graph.id_to_row or self.tomb.is_tombstoned(n))
        if dead / len(layer0) >= self.tombstone_prune_fraction:
            self.repair_engine.repair_node(u, self.graph, self.tomb)
            self.dirty.remove(u)
            self.stats.repair_count += 1

    # ------------------------------------------------------------------ #
    # core operations
    # ------------------------------------------------------------------ #
    def insert(self, node_id: IdT, vec: np.ndarray) -> None:
        vec = preprocess(np.asarray(vec, dtype=np.float32).reshape(-1), self.metric)
        self._orig_vectors[node_id] = vec.copy()
        if node_id in self.tomb:
            self.tomb.discard(node_id)
        self.graph.add_node(node_id, vec, is_tombstoned=self.tomb.is_tombstoned)
        self.stats.insert_count += 1

    def remove(self, node_id: IdT) -> None:
        if node_id not in self.graph.id_to_row or self.tomb.is_tombstoned(node_id):
            return
        self.tomb.add(node_id)
        if node_id == self.graph.entry_point:
            self._replace_entry_point()
        self._mark_neighbors_dirty(node_id)
        self.stats.delete_count += 1

    def update(self, node_id: IdT, vec: np.ndarray) -> None:
        # delete old version, reinsert new vector under a fresh internal id but
        # keep the external id stable (version semantics)
        if node_id in self.graph.id_to_row and not self.tomb.is_tombstoned(node_id):
            self.tomb.add(node_id)
            if node_id == self.graph.entry_point:
                self._replace_entry_point()
            self._mark_neighbors_dirty(node_id)
        new_internal = self._fresh_version_id(node_id)
        vec = preprocess(np.asarray(vec, dtype=np.float32).reshape(-1), self.metric)
        self._orig_vectors[new_internal] = vec.copy()
        self.graph.add_node(new_internal, vec, is_tombstoned=self.tomb.is_tombstoned)
        self._version_alias[new_internal] = node_id
        self.stats.update_count += 1

    def _fresh_version_id(self, external_id: IdT) -> IdT:
        # negative, high-offset internal id to avoid colliding with user ids
        self._version_counter += 1
        return -(10_000_000 + self._version_counter)

    def _replace_entry_point(self) -> None:
        """Find a new top-level live entry point after the old one is tombstoned."""
        best_id = None
        best_level = -1
        for nid in self.graph.id_to_row:
            if self.tomb.is_tombstoned(nid):
                continue
            lv = self.graph.level.get(nid, 0)
            if lv > best_level:
                best_level = lv
                best_id = nid
        self.graph.entry_point = best_id
        self.graph.max_level = best_level if best_id is not None else -1

    def search(self, query: np.ndarray, k: int = 10, ef_search: int = 50) -> List[Tuple[float, IdT]]:
        if self.graph.entry_point is None:
            return []
        q = preprocess(np.asarray(query, dtype=np.float32).reshape(-1), self.metric)
        ep = [self.graph.entry_point]
        for lc in range(self.graph.max_level, 0, -1):
            res = self.graph.search_layer(q, ep, ef=1, layer=lc,
                                          is_tombstoned=self.tomb.is_tombstoned,
                                          skip_tombstones=True)
            if res:
                ep = [res[0][1]]
        res = self.graph.search_layer(
            q, ep, ef=max(ef_search, k), layer=0,
            is_tombstoned=self.tomb.is_tombstoned, collect_live_only=True,
            skip_tombstones=True,
        )
        # map version-aliased ids back to external ids, drop tombstoned
        out = []
        for d, nid in res:
            if self.tomb.is_tombstoned(nid):
                continue
            if self._hotness:  # record access frequency of retrieved data items
                self.visit_count[nid] = self.visit_count.get(nid, 0) + 1
            ext = getattr(self, "_version_alias", {}).get(nid, nid)
            out.append((d, ext))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ #
    # maintenance
    # ------------------------------------------------------------------ #
    def repair(self, budget: int = -1) -> int:
        """Repair up to ``budget`` dirty nodes (priority-ordered). -1 = all."""
        if budget == -1:
            budget = self.dirty.size()
        if budget <= 0:
            return 0
        nodes = self.dirty.pop_top_k(budget)
        if not nodes:
            return 0
        self.repair_engine.repair_batch(nodes, self.graph, self.tomb)
        self.stats.repair_count += len(nodes)
        return len(nodes)

    def should_rebuild(self) -> bool:
        n_live = self.graph.n_nodes - len(self.tomb)
        if n_live <= 0:
            return False
        if len(self.tomb) / max(self.graph.n_nodes, 1) > self.tombstone_threshold:
            return True
        if self.dirty.avg_deficit() > self.degree_deficit_threshold * self.graph.M0:
            return True
        return False

    def rebuild(self) -> None:
        """Full rebuild from live vectors only (tombstones discarded)."""
        live_items = [
            (getattr(self, "_version_alias", {}).get(nid, nid), self._orig_vectors[nid])
            for nid in self.graph.id_to_row
            if not self.tomb.is_tombstoned(nid) and nid in self._orig_vectors
        ]
        self.graph = HNSWGraph(
            self.dim, M=self.M, ef_construction=self.ef_construction,
            metric=self.metric, seed=self.seed,
        )
        self.tomb = TombstoneManager()
        policy = self.dirty.policy
        self.dirty = DirtySet(policy=policy, seed=self.seed)
        new_orig: Dict[IdT, np.ndarray] = {}
        self._version_alias = {}
        order = list(range(len(live_items)))
        self._rng.shuffle(order)
        for i in order:
            ext_id, vec = live_items[i]
            new_orig[ext_id] = vec
            self.graph.add_node(ext_id, vec, is_tombstoned=self.tomb.is_tombstoned)
        self._orig_vectors = new_orig
        self.stats.rebuild_count += 1

    # ------------------------------------------------------------------ #
    # stats / recall estimation
    # ------------------------------------------------------------------ #
    def get_stats(self) -> IndexStats:
        n_total = self.graph.n_nodes
        n_tomb = len(self.tomb)
        self.stats.n_live = n_total - n_tomb
        self.stats.n_tombstone = n_tomb
        self.stats.n_dirty = self.dirty.size()
        self.stats.tombstone_ratio = n_tomb / max(n_total, 1)
        self.stats.avg_degree_deficit = self.dirty.avg_deficit()
        self.stats.dist_calls = self.graph.dist_calls
        return self.stats

    def live_vectors(self) -> Tuple[np.ndarray, List[IdT]]:
        """Return (matrix, external_ids) of all current live vectors."""
        ids, rows = [], []
        for nid in self.graph.id_to_row:
            if self.tomb.is_tombstoned(nid):
                continue
            ext = getattr(self, "_version_alias", {}).get(nid, nid)
            ids.append(ext)
            rows.append(self._orig_vectors[nid])
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32), []
        return np.vstack(rows).astype(np.float32), ids

    def estimate_recall(self, n_probe: int = 100, k: int = 10, ef_search: int = 50) -> float:
        """Cheap internal recall estimate via probe queries against exact kNN
        over the current live set (used by the adaptive trigger, not for reported
        numbers -- those use the held-out query set)."""
        mat, ids = self.live_vectors()
        if len(ids) <= k:
            return 1.0
        n_probe = min(n_probe, len(ids))
        probe_rows = self._rng.choice(len(ids), size=n_probe, replace=False)
        hits = 0
        for r in probe_rows:
            q = mat[r]
            d = np.einsum("ij,ij->i", mat - q, mat - q)
            gt = set(np.argsort(d)[: k + 1].tolist()) - {int(r)}
            gt_ids = {ids[i] for i in list(gt)[:k]}
            res = self.search(q, k=k, ef_search=ef_search)
            pred_ids = {i for _, i in res}
            hits += len(gt_ids & pred_ids)
        return hits / (n_probe * k)
