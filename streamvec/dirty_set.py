"""DirtySet: priority queue scheduling repairs by degree deficit.

This is StreamVec's main scheduling heuristic.  A node becomes *dirty* when it
loses live neighbors to tombstones.  Its priority is the **degree deficit**
``deficit = M_target - live_neighbor_count`` -- the number of neighbor slots it
needs refilled.  Repairing high-deficit nodes first targets exactly the nodes
whose local navigability (and hence recall contribution) has degraded most.

A binary max-heap gives O(log n) push and pop.  Because Python's ``heapq`` has
no decrease/increase-key, we use **lazy deletion**: the authoritative deficit
lives in ``_deficit`` and stale heap entries are skipped on pop.  Alternative
scheduling policies (FIFO, Random, EagerAll) are provided for the ablation.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Dict, List, Optional

import numpy as np

IdT = int


class DirtySet:
    POLICIES = ("priority", "fifo", "random", "eager")

    def __init__(self, policy: str = "priority", seed: int = 42) -> None:
        if policy not in self.POLICIES:
            raise ValueError(f"unknown policy {policy!r}, must be one of {self.POLICIES}")
        self.policy = policy
        self._deficit: Dict[IdT, int] = {}      # authoritative current deficit
        self._heap: List = []                    # (-deficit, counter, id) max-heap
        self._counter = itertools.count()
        self._fifo: deque[IdT] = deque()
        self._members: set[IdT] = set()
        self._rng = np.random.default_rng(seed)
        self._priority: Dict[IdT, float] = {}  # heap key (== deficit unless hotness-weighted)

    # ------------------------------------------------------------------ #
    def mark_dirty(self, node_id: IdT, deficit: int, priority: float = None) -> None:
        """Record/refresh a node's degree deficit and (optionally) a separate
        heap priority.  ``priority`` defaults to ``deficit``; the hotness-aware
        scheduler passes ``deficit + alpha*visits`` while keeping ``deficit`` for
        health statistics."""
        if deficit <= 0:
            # node is healthy: drop it
            if node_id in self._members:
                self._members.discard(node_id)
                self._deficit.pop(node_id, None)
                self._priority.pop(node_id, None)
            return
        pri = float(deficit if priority is None else priority)
        self._deficit[node_id] = deficit
        self._priority[node_id] = pri
        if node_id not in self._members:
            self._members.add(node_id)
            self._fifo.append(node_id)
        # always push a fresh heap entry; stale ones are filtered on pop
        heapq.heappush(self._heap, (-pri, next(self._counter), node_id))

    def update_deficit(self, node_id: IdT, new_deficit: int) -> None:
        self.mark_dirty(node_id, new_deficit)

    def remove(self, node_id: IdT) -> None:
        self._members.discard(node_id)
        self._deficit.pop(node_id, None)
        self._priority.pop(node_id, None)

    def deficit_of(self, node_id: IdT) -> int:
        return self._deficit.get(node_id, 0)

    def __contains__(self, node_id: IdT) -> bool:
        return node_id in self._members

    def size(self) -> int:
        return len(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def avg_deficit(self) -> float:
        if not self._deficit:
            return 0.0
        return float(np.mean(list(self._deficit.values())))

    def total_deficit(self) -> int:
        return int(sum(self._deficit.values()))

    # ------------------------------------------------------------------ #
    def pop_top_k(self, k: int) -> List[IdT]:
        """Return up to k dirty node ids according to the scheduling policy."""
        if k <= 0:
            return []
        if self.policy == "eager":
            k = len(self._members)
        if self.policy == "priority":
            return self._pop_priority(k)
        if self.policy == "fifo":
            return self._pop_fifo(k)
        # random
        return self._pop_random(k)

    def _pop_priority(self, k: int) -> List[IdT]:
        out: List[IdT] = []
        while self._heap and len(out) < k:
            neg_def, _, nid = heapq.heappop(self._heap)
            if nid not in self._members:
                continue  # stale (already repaired/removed)
            if -neg_def != self._priority.get(nid):
                continue  # stale priority; a fresher entry exists
            out.append(nid)
            self.remove(nid)
        return out

    def _pop_fifo(self, k: int) -> List[IdT]:
        out: List[IdT] = []
        while self._fifo and len(out) < k:
            nid = self._fifo.popleft()
            if nid in self._members:
                out.append(nid)
                self.remove(nid)
        return out

    def _pop_random(self, k: int) -> List[IdT]:
        members = list(self._members)
        if not members:
            return []
        k = min(k, len(members))
        chosen = self._rng.choice(len(members), size=k, replace=False)
        out = [members[int(i)] for i in chosen]
        for nid in out:
            self.remove(nid)
        return out
