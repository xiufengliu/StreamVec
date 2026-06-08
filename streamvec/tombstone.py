"""Tombstone management for lazy deletion."""
from __future__ import annotations

from typing import Set

IdT = int


class TombstoneManager:
    """Tracks soft-deleted (tombstoned) node ids.

    Lazy deletion: ``add`` is O(1); edges incident to a tombstoned node are not
    rewired until the repair engine processes the affected (dirty) neighbors.
    """

    def __init__(self) -> None:
        self._tombstoned: Set[IdT] = set()

    def add(self, node_id: IdT) -> None:
        self._tombstoned.add(node_id)

    def discard(self, node_id: IdT) -> None:
        self._tombstoned.discard(node_id)

    def is_tombstoned(self, node_id: IdT) -> bool:
        return node_id in self._tombstoned

    def __contains__(self, node_id: IdT) -> bool:
        return node_id in self._tombstoned

    def __len__(self) -> int:
        return len(self._tombstoned)

    @property
    def ids(self) -> Set[IdT]:
        return self._tombstoned
