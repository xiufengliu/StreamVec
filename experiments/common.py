"""Shared experiment harness: builds systems, drives streams, measures metrics.

A ``SystemAdapter`` wraps either a StreamVecIndex or a baseline behind a uniform
streaming interface so every experiment treats them identically.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamvec import StreamVecIndex
from streamvec.distance import DistanceMetric
from streamvec.baselines import HNSWAppend, HNSWRebuild, FAISSIVFRebuild, PyRebuild
from streamvec.config import ExperimentConfig
from streamvec.groundtruth import compute_groundtruth, recall_at_k, ndcg_at_k
from streamvec.metrics import percentiles, estimate_index_memory_mb

METRIC_MAP = {"l2": DistanceMetric.L2, "cosine": DistanceMetric.COSINE, "ip": DistanceMetric.IP}


class SystemAdapter:
    """Uniform streaming interface over StreamVec or a baseline."""

    def __init__(self, kind: str, cfg: ExperimentConfig, repair_policy: Optional[str] = None):
        self.kind = kind
        self.cfg = cfg
        self.repair_policy = repair_policy or cfg.repair_policy
        self._build()

    def _build(self):
        cfg = self.cfg
        dim = cfg.dim
        self.no_repair = False
        if self.kind in ("streamvec", "streamvec_norepair"):
            self.no_repair = self.kind == "streamvec_norepair"
            self.sys = StreamVecIndex(
                dim=dim, M=cfg.M, ef_construction=cfg.ef_construction,
                metric=METRIC_MAP[cfg.metric], repair_policy=self.repair_policy,
                ef_repair=cfg.ef_repair, tombstone_threshold=cfg.tombstone_threshold,
                degree_deficit_threshold=cfg.degree_deficit_threshold, seed=cfg.seed,
            )
            self.is_sv = True
        elif self.kind == "hnsw_append":
            self.sys = HNSWAppend(dim=dim, M=cfg.M, ef_construction=cfg.ef_construction, metric=cfg.metric)
            self.is_sv = False
        elif self.kind == "hnsw_rebuild":
            self.sys = HNSWRebuild(dim=dim, M=cfg.M, ef_construction=cfg.ef_construction,
                                   rebuild_interval=cfg.rebuild_interval, metric=cfg.metric)
            self.is_sv = False
        elif self.kind == "faiss_ivf":
            self.sys = FAISSIVFRebuild(dim=dim, rebuild_interval=cfg.rebuild_interval, metric=cfg.metric)
            self.is_sv = False
        elif self.kind == "py_rebuild":
            self.sys = PyRebuild(dim=dim, M=cfg.M, ef_construction=cfg.ef_construction,
                                 rebuild_interval=cfg.rebuild_interval, metric=cfg.metric)
            self.is_sv = False
        else:
            raise ValueError(f"unknown system kind {self.kind!r}")
        self._since_repair = 0
        self.rebuilds = 0

    @property
    def name(self) -> str:
        if self.is_sv:
            if self.no_repair:
                return "StreamVec-NoRepair"
            return f"StreamVec-{self.repair_policy}" if self.repair_policy != "priority" else "StreamVec"
        return self.sys.name

    def apply(self, op: str, vid: int, vec) -> None:
        if op == "insert":
            self.sys.insert(vid, vec)
        elif op == "delete":
            self.sys.remove(vid)
        else:
            self.sys.update(vid, vec)

    def maintain(self, op_count: int) -> None:
        if self.is_sv:
            if self.no_repair:
                return  # lazy deletion only -- no repair, no rebuild (control)
            self._since_repair += 1
            if self._since_repair >= self.cfg.repair_interval:
                self.sys.repair(budget=self.cfg.repair_budget)
                self._since_repair = 0
                if self.sys.should_rebuild():
                    self.sys.rebuild()
                    self.rebuilds += 1
        else:
            self.sys.maybe_maintain(op_count)

    def finalize(self) -> None:
        if self.is_sv:
            if not self.no_repair:
                self.sys.repair(budget=-1)
        elif hasattr(self.sys, "_build"):
            self.sys._build()

    def search(self, q, k, ef):
        return self.sys.search(q, k=k, ef_search=ef)

    def live_vectors(self):
        return self.sys.live_vectors()

    def memory_mb(self) -> float:
        if self.is_sv:
            return estimate_index_memory_mb(self.sys)
        mat, ids = self.sys.live_vectors()
        return mat.nbytes / (1024 * 1024) * 2  # vectors + index structures (approx)


def measure_recall(adapter: SystemAdapter, queries: np.ndarray, k: int, ef: int) -> Tuple[float, Dict]:
    """Recall@k against exact ground truth over the current live set + latency.

    For the cosine metric the true neighbors are defined on unit-normalized
    vectors, so ground truth is computed on normalized live vectors/queries to
    match every system's internal cosine search (otherwise a raw-L2 ground truth
    would mismatch a cosine index and understate its recall).
    """
    mat, ids = adapter.live_vectors()
    if len(ids) <= k:
        return 1.0, {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "qps": 0}
    if adapter.cfg.metric == "cosine":
        from streamvec.distance import normalize
        gt_mat = normalize(mat)
        gt_q = normalize(queries)
    else:
        gt_mat, gt_q = mat, queries
    gt = compute_groundtruth(gt_mat, ids, gt_q, k=k)
    preds, lats = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = adapter.search(q, k=k, ef=ef)
        lats.append((time.perf_counter() - t0) * 1000)
        preds.append([i for _, i in res])
    r = recall_at_k(preds, gt, k=k)
    pc = percentiles(lats)
    pc["qps"] = len(queries) / (sum(lats) / 1000.0) if sum(lats) > 0 else 0.0
    return r, pc


def run_stream(
    adapter: SystemAdapter,
    seed_inserts: List[Tuple[int, np.ndarray]],
    ops: List[Tuple[str, int, object]],
    queries: np.ndarray,
    cfg: ExperimentConfig,
    eval_ef: Optional[int] = None,
    checkpoints: bool = True,
) -> Dict:
    """Seed the index, replay the stream, and record per-checkpoint + final metrics."""
    eval_ef = eval_ef or cfg.ef_search
    t0 = time.perf_counter()
    for vid, vec in seed_inserts:
        adapter.apply("insert", vid, vec)
    seed_time = time.perf_counter() - t0

    history: List[Dict] = []
    t_stream = time.perf_counter()
    for n, (op, vid, vec) in enumerate(ops, 1):
        adapter.apply(op, vid, vec)
        adapter.maintain(n)
        if checkpoints and cfg.eval_every and n % cfg.eval_every == 0:
            r, pc = measure_recall(adapter, queries, cfg.k, eval_ef)
            history.append({"ops": n, "recall": r, **pc,
                            "elapsed_s": time.perf_counter() - t_stream})
    adapter.finalize()
    stream_time = time.perf_counter() - t_stream

    r, pc = measure_recall(adapter, queries, cfg.k, eval_ef)
    n_ops = len(ops)
    result = {
        "system": adapter.name,
        "n_ops": n_ops,
        "seed_time_s": seed_time,
        "stream_time_s": stream_time,
        "update_throughput_ops_s": n_ops / stream_time if stream_time > 0 else 0.0,
        "final_recall": r,
        "memory_mb": adapter.memory_mb(),
        "rebuilds": adapter.rebuilds if adapter.is_sv else getattr(adapter.sys, "rebuild_count", 0),
        **{f"query_{kk}": vv for kk, vv in pc.items()},
    }
    if adapter.is_sv:
        st = adapter.sys.get_stats()
        result.update({
            "tombstone_ratio": st.tombstone_ratio,
            "n_live": st.n_live, "n_tombstone": st.n_tombstone,
            "repair_count": st.repair_count, "dist_calls": st.dist_calls,
        })
    return {"result": result, "history": history}
