"""Experiment configuration dataclasses with YAML/JSON load-save."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    name: str = "exp"
    dataset: str = "sift"            # 'sift' | 'synthetic'
    n_base: int = 20000
    n_query: int = 500
    dim: int = 128                  # only used for synthetic
    metric: str = "l2"
    # index params
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    k: int = 10
    # streaming params
    n_ops: int = 40000
    insert_ratio: float = 0.6
    delete_ratio: float = 0.2
    update_ratio: float = 0.2
    seed_fraction: float = 0.5
    # StreamVec maintenance params
    repair_policy: str = "priority"
    ef_repair: int = 50
    repair_budget: int = 64          # dirty nodes repaired per repair_interval ops
    repair_interval: int = 256       # call repair() every N stream ops
    tombstone_threshold: float = 0.15
    degree_deficit_threshold: float = 0.3
    rebuild_interval: int = 50000    # baseline rebuild cadence
    # eval cadence
    eval_every: int = 5000
    seed: int = 42

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**json.load(f))
