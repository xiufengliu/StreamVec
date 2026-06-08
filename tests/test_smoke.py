"""Smoke / sanity tests for StreamVec. Run: python -m tests.test_smoke"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamvec import StreamVecIndex, DistanceMetric
from streamvec.datasets import make_synthetic
from streamvec.stream_gen import generate_stream, seed_ids
from streamvec.groundtruth import compute_groundtruth, recall_at_k
from streamvec.baselines import HNSWAppend, HNSWRebuild, FAISSIVFRebuild


def brute_force_knn(base, ids, queries, k=10):
    out = []
    for q in queries:
        d = np.einsum("ij,ij->i", base - q, base - q)
        order = np.argsort(d)[:k]
        out.append([ids[i] for i in order])
    return out


def test_static_recall():
    base, queries = make_synthetic(n_base=1500, dim=64, n_query=80, seed=1)
    idx = StreamVecIndex(dim=64, M=16, ef_construction=80, seed=1)
    for i, v in enumerate(base):
        idx.insert(i, v)
    gt = brute_force_knn(base, list(range(len(base))), queries, k=10)
    pred = [[i for _, i in idx.search(q, k=10, ef_search=64)] for q in queries]
    r = recall_at_k(pred, gt, k=10)
    print(f"[static_recall] StreamVec Recall@10 = {r:.4f}  (dist_calls={idx.graph.dist_calls})")
    assert r > 0.90, f"static recall too low: {r}"
    return r


def test_streaming_and_repair():
    base, queries = make_synthetic(n_base=3000, dim=64, n_query=80, seed=2)
    idx = StreamVecIndex(dim=64, M=16, ef_construction=80, repair_policy="priority", seed=2)
    sids = seed_ids(base, 0.5)
    for i in sids:
        idx.insert(i, base[i])
    ops, _ = generate_stream(base, n_ops=3000, seed=2)
    t0 = time.perf_counter()
    for n, (op, vid, vec) in enumerate(ops, 1):
        if op == "insert":
            idx.insert(vid, vec)
        elif op == "delete":
            idx.remove(vid)
        else:
            idx.update(vid, vec)
        if n % 256 == 0:
            idx.repair(budget=64)
    idx.repair(budget=-1)
    dt = time.perf_counter() - t0
    st = idx.get_stats()
    print(f"[streaming] {len(ops)} ops in {dt:.2f}s ({len(ops)/dt:.0f} ops/s); "
          f"live={st.n_live} tomb={st.n_tombstone} dirty={st.n_dirty} "
          f"tomb_ratio={st.tombstone_ratio:.3f}")
    # build ground truth on current live set
    mat, ids = idx.live_vectors()
    gt = compute_groundtruth(mat, ids, queries, k=10)
    pred = [[i for _, i in idx.search(q, k=10, ef_search=64)] for q in queries]
    r = recall_at_k(pred, gt, k=10)
    print(f"[streaming] post-stream Recall@10 = {r:.4f}")
    # every returned id must be a currently-live external id (deleted docs and
    # stale pre-update versions must never surface)
    live_set = set(ids)
    flat = {i for row in pred for i in row}
    stale = flat - live_set
    assert not stale, f"stale/deleted ids leaked into results: {list(stale)[:5]}"
    assert r > 0.80, f"streaming recall too low: {r}"
    return r


def test_baselines():
    base, queries = make_synthetic(n_base=2000, dim=64, n_query=60, seed=3)
    for B in (HNSWAppend, HNSWRebuild, FAISSIVFRebuild):
        b = B(dim=64, M=16, ef_construction=80, rebuild_interval=1000, metric="l2")
        for i, v in enumerate(base):
            b.insert(i, v)
            b.maybe_maintain(i)
        if hasattr(b, "_build"):
            b._build()
        res = b.search(queries[0], k=10, ef_search=64)
        print(f"[baseline] {b.name}: returned {len(res)} results")
        assert len(res) > 0


if __name__ == "__main__":
    test_static_recall()
    test_streaming_and_repair()
    test_baselines()
    print("\nALL SMOKE TESTS PASSED")
