"""Mixed-operation stream workload generator (blueprint Part 4.2 / Prompt 3)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

Op = Tuple[str, int, Optional[np.ndarray]]


def generate_stream(
    base_vectors: np.ndarray,
    n_ops: int,
    insert_ratio: float = 0.6,
    delete_ratio: float = 0.2,
    update_ratio: float = 0.2,
    seed_fraction: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Op], List[int]]:
    """Generate a realistic mixed insert/delete/update stream.

    The index is seeded with ``seed_fraction`` of ``base_vectors`` (these are
    returned as initial inserts at the front of the op list is *not* done here --
    callers seed separately via ``seed_ids``).  Returns ``(ops, live_ids)`` where
    ops is a list of ``(op_type, id, vector_or_None)`` and live_ids is the final
    set of live ids.  Deletes/updates only ever target currently-live ids.
    """
    assert abs(insert_ratio + delete_ratio + update_ratio - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    n = len(base_vectors)
    n_seed = int(n * seed_fraction)
    live_ids = list(range(n_seed))
    next_id = n_seed
    ops: List[Op] = []

    for _ in range(n_ops):
        r = rng.random()
        if r < insert_ratio or len(live_ids) < 100:
            vec = base_vectors[next_id % n]
            ops.append(("insert", next_id, vec))
            live_ids.append(next_id)
            next_id += 1
        elif r < insert_ratio + delete_ratio and len(live_ids) > 100:
            j = int(rng.integers(len(live_ids)))
            vid = live_ids.pop(j)
            ops.append(("delete", vid, None))
        else:
            j = int(rng.integers(len(live_ids)))
            vid = live_ids[j]
            new_vec = base_vectors[int(rng.integers(n))]
            ops.append(("update", vid, new_vec))
    return ops, live_ids


def seed_ids(base_vectors: np.ndarray, seed_fraction: float = 0.5) -> List[int]:
    return list(range(int(len(base_vectors) * seed_fraction)))
