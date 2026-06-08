"""Index checkpointing (save/load full StreamVec state via pickle)."""
from __future__ import annotations

import os
import pickle


def save_index(index, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
