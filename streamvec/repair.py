"""RepairEngine: refills the neighbor lists of dirty nodes with live nodes.

For a dirty node ``u`` the engine (1) collects ``u``'s surviving live neighbors,
(2) runs a localized beam search seeded from those live neighbors to find
replacement candidates, (3) applies HNSW heuristic neighbor selection to refill
``u``'s neighbor list up to the per-layer target degree, and (4) adds reverse
edges.  Seeding the beam search from ``u``'s own neighbors (rather than the
global entry point) is the "reuse candidates across nearby dirty nodes"
amortization -- it keeps each repair localized and cheap.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .hnsw_graph import HNSWGraph
from .tombstone import TombstoneManager

IdT = int


class RepairEngine:
    def __init__(self, ef_repair: int = 50) -> None:
        self.ef_repair = ef_repair
        self.repaired_nodes = 0
        self.repair_dist_calls = 0

    # ------------------------------------------------------------------ #
    def _candidates_for_layer(
        self,
        node_id: IdT,
        layer: int,
        graph: HNSWGraph,
        tomb: TombstoneManager,
        ef: int,
    ) -> List[Tuple[float, IdT]]:
        vec = graph.vector(node_id)
        nbrs = graph.neighbors[node_id][layer]
        live_nbrs = [n for n in nbrs if n in graph.id_to_row and not tomb.is_tombstoned(n)]
        # entry points: live neighbors, else fall back to the global entry point
        if live_nbrs:
            entry = live_nbrs
        elif graph.entry_point is not None:
            entry = [graph.entry_point]
        else:
            return []
        # two-hop seeds: neighbors-of-neighbors broaden the candidate pool cheaply
        seeds = set(entry)
        for ln in live_nbrs:
            ln_nbrs = graph.neighbors.get(ln)
            if ln_nbrs and layer < len(ln_nbrs):
                for nn in ln_nbrs[layer]:
                    if nn in graph.id_to_row and not tomb.is_tombstoned(nn):
                        seeds.add(nn)
        before = graph.dist_calls
        cand = graph.search_layer(
            vec,
            list(seeds),
            ef=ef,
            layer=layer,
            is_tombstoned=tomb.is_tombstoned,
            collect_live_only=True,
            skip_tombstones=True,
        )
        self.repair_dist_calls += graph.dist_calls - before
        # exclude self
        return [(d, c) for d, c in cand if c != node_id]

    def repair_node(
        self,
        node_id: IdT,
        graph: HNSWGraph,
        tomb: TombstoneManager,
        ef_repair: int = None,
    ) -> int:
        """Repair a single dirty node across all layers it participates in.

        Returns the number of new edges added.
        """
        if node_id not in graph.id_to_row or tomb.is_tombstoned(node_id):
            return 0
        ef = ef_repair or self.ef_repair
        node_level = graph.level.get(node_id, 0)
        added = 0
        for layer in range(node_level, -1, -1):
            m_target = graph.M0 if layer == 0 else graph.M
            nbrs = graph.neighbors[node_id][layer]
            live = [n for n in nbrs if n in graph.id_to_row and not tomb.is_tombstoned(n)]
            if len(live) >= m_target:
                # purge dead edges but no refill needed
                graph.neighbors[node_id][layer] = live
                continue
            cand = self._candidates_for_layer(node_id, layer, graph, tomb, ef)
            # combine with surviving live neighbors
            vec = graph.vector(node_id)
            have = set(live)
            pool: List[Tuple[float, IdT]] = []
            for n in live:
                dn = graph._dist_to_many(vec, [n])[0]
                pool.append((float(dn), n))
            for d, c in cand:
                if c not in have:
                    have.add(c)
                    pool.append((d, c))
            chosen = graph.select_neighbors_heuristic(node_id, pool, m_target)
            graph.neighbors[node_id][layer] = list(chosen)
            # reverse edges (keep the live subgraph symmetric & navigable)
            for nb in chosen:
                if nb == node_id:
                    continue
                nb_level = graph.level.get(nb, 0)
                if layer <= nb_level and node_id not in graph.neighbors[nb][layer]:
                    graph.neighbors[nb][layer].append(node_id)
                    graph._prune_if_needed(nb, layer)
                    added += 1
        self.repaired_nodes += 1
        return added

    def repair_batch(
        self,
        dirty_nodes: List[IdT],
        graph: HNSWGraph,
        tomb: TombstoneManager,
        ef_repair: int = None,
    ) -> int:
        """Repair a batch of dirty nodes.  Processing nearby nodes consecutively
        lets each repair reuse the warmed-up local neighborhood."""
        total = 0
        for nid in dirty_nodes:
            total += self.repair_node(nid, graph, tomb, ef_repair)
        return total
