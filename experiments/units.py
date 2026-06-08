"""Resumable, unit-based experiment definitions.

The sandbox terminates background processes when a shell call ends, and each
call is time-limited.  So every experiment is decomposed into independent
*units* (one system / policy / budget / scale point).  ``run_unit.py`` runs as
many pending units as fit in a wall-clock budget, appends them to the
experiment's CSV, and exits; re-invoking resumes with the remaining units.

Scales are deliberately modest to fit the sandbox (4 cores / 3.8 GB, pure-Python
index); they are reported honestly in the paper.  Determinism: every unit
regenerates the same seeded stream, so partial runs are reproducible.
"""
from __future__ import annotations

import os, sys, time, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import SystemAdapter, run_stream, measure_recall
from streamvec.config import ExperimentConfig
from streamvec.datasets import load_dataset, make_synthetic
from streamvec.stream_gen import generate_stream, seed_ids
from streamvec.groundtruth import compute_groundtruth, recall_at_k, ndcg_at_k
from streamvec.metrics import estimate_index_memory_mb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def _csv_path(exp, name):
    return os.path.join(RES, exp, name)


def _append_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    exists = os.path.exists(path)
    # if header differs, rewrite by merging keys
    old = []
    if exists:
        with open(path) as f:
            rd = csv.DictReader(f)
            old = list(rd)
            for k in (rd.fieldnames or []):
                if k not in keys:
                    keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in old + rows:
            w.writerow(r)


def _done_units(path, key):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if key in r:
                out.add(str(r[key]))
    return out


# ======================================================================= #
# EXP1 -- throughput/recall Pareto                                         #
# ======================================================================= #
EXP1_SYSTEMS = ["streamvec", "py_rebuild", "hnsw_append", "hnsw_rebuild", "faiss_ivf"]
EXP1_EF = [10, 20, 40, 80, 160]


def _exp1_cfg():
    return ExperimentConfig(name="exp1", dataset="sift", n_base=8000, n_query=400, metric="l2",
                            M=16, ef_construction=100, k=10, n_ops=6000,
                            insert_ratio=0.6, delete_ratio=0.2, update_ratio=0.2,
                            repair_policy="priority", repair_budget=64, repair_interval=256,
                            tombstone_threshold=0.25, rebuild_interval=4000, eval_every=0, seed=42)


def exp1_units():
    return [{"system": s} for s in EXP1_SYSTEMS]


def exp1_pending():
    done = _done_units(_csv_path("exp1", "pareto.csv"), "system")
    name_map = {"streamvec": "StreamVec", "py_rebuild": "Python-Rebuild",
                "hnsw_append": "HNSW-Append", "hnsw_rebuild": "HNSW-Rebuild",
                "faiss_ivf": "FAISS-IVF-Rebuild"}
    return [u for u in exp1_units() if name_map[u["system"]] not in done]


def exp1_run_one(unit):
    cfg = _exp1_cfg()
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, cfg.seed_fraction)
    seed_inserts = [(i, base[i]) for i in sids]
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=cfg.seed_fraction, seed=cfg.seed)
    ad = SystemAdapter(unit["system"], cfg)
    out = run_stream(ad, seed_inserts, ops, queries, cfg, eval_ef=cfg.ef_search, checkpoints=False)
    br = out["result"]
    rows = []
    for ef in EXP1_EF:
        r, pc = measure_recall(ad, queries, cfg.k, ef)
        rows.append({"system": ad.name, "ef_search": ef, "recall_at_10": round(r, 4),
                     "query_qps": round(pc["qps"], 1), "p99_ms": round(pc["p99_ms"], 3),
                     "mean_ms": round(pc["mean_ms"], 3),
                     "update_throughput_ops_s": round(br["update_throughput_ops_s"], 1),
                     "memory_mb": round(br["memory_mb"], 2), "rebuilds": br["rebuilds"]})
    _append_rows(_csv_path("exp1", "pareto.csv"), rows)
    return f"exp1:{ad.name}"


# ======================================================================= #
# EXP2 -- recall vs tombstone ratio                                        #
# ======================================================================= #
EXP2_TOMB = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def _exp2_cfg():
    # M=8 (sparse, memory-lean HNSW setting) + low ef operating point: the
    # regime where the live graph is sensitive to fragmentation, so repair is
    # measurable.  At M=16 HNSW edge redundancy makes recall churn-robust
    # regardless (reported as a scope finding in the paper).
    return ExperimentConfig(name="exp2", dataset="sift", n_base=7000, n_query=400, metric="l2",
                            M=8, ef_construction=80, k=10, ef_search=20,
                            repair_policy="priority", repair_budget=16, ef_repair=48, seed=42)


def exp2_pending():
    done = _done_units(_csv_path("exp2", "recall_vs_tombstone.csv"), "kind")
    return [{"kind": k} for k in ("streamvec", "streamvec_norepair", "hnsw_append") if k not in done]


def exp2_run_one(unit):
    cfg = _exp2_cfg()
    kind = unit["kind"]
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    rng = np.random.default_rng(cfg.seed)
    delete_order = rng.permutation(cfg.n_base)
    ad = SystemAdapter(kind, cfg)
    for i in range(cfg.n_base):
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    deleted = 0
    rows = []
    for lvl in EXP2_TOMB:
        target = int(lvl * cfg.n_base)
        while deleted < target:
            ad.apply("delete", int(delete_order[deleted]), None)
            deleted += 1
            # amortized repair during deletion (StreamVec), mirroring streaming
            if kind == "streamvec" and deleted % 256 == 0:
                ad.sys.repair(budget=cfg.repair_budget)
        if kind == "streamvec":
            ad.sys.repair(budget=cfg.repair_budget)  # drain remaining high-deficit nodes
        r, pc = measure_recall(ad, queries, cfg.k, cfg.ef_search)
        rows.append({"kind": kind, "system": ad.name, "tombstone_ratio": lvl,
                     "recall_at_10": round(r, 4), "query_qps": round(pc["qps"], 1)})
        print(f"  exp2 {kind} tomb={lvl:.2f} recall={r:.3f}", flush=True)
    _append_rows(_csv_path("exp2", "recall_vs_tombstone.csv"), rows)
    return f"exp2:{ad.name}"


# ======================================================================= #
# EXP3 -- repair scheduling ablation                                       #
# ======================================================================= #
EXP3_POLICIES = ["priority", "fifo", "random", "eager", "norepair"]


def _exp3_cfg():
    # sparse graph + scarce repair budget + delete-heavy stream + low ef =>
    # the dirty set backs up and scheduling ORDER matters.
    return ExperimentConfig(name="exp3", dataset="sift", n_base=8000, n_query=400, metric="l2",
                            M=8, ef_construction=80, k=10, ef_search=20, n_ops=7000,
                            insert_ratio=0.35, delete_ratio=0.5, update_ratio=0.15,
                            repair_budget=8, repair_interval=128, ef_repair=48,
                            tombstone_threshold=0.99, eval_every=1750, seed=42)


def exp3_pending():
    done = _done_units(_csv_path("exp3", "ablation_final.csv"), "policy")
    return [{"policy": p} for p in EXP3_POLICIES if p not in done]


def exp3_run_one(unit):
    cfg = _exp3_cfg()
    policy = unit["policy"]
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, cfg.seed_fraction)
    seed_inserts = [(i, base[i]) for i in sids]
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=cfg.seed_fraction, seed=cfg.seed)
    if policy == "norepair":
        ad = SystemAdapter("streamvec_norepair", cfg)
    else:
        ad = SystemAdapter("streamvec", cfg, repair_policy=policy)
    out = run_stream(ad, seed_inserts, ops, queries, cfg, eval_ef=cfg.ef_search, checkpoints=True)
    res = out["result"]
    res["policy"] = policy
    res["repair_dist_calls"] = ad.sys.repair_engine.repair_dist_calls
    res["final_recall"] = round(res["final_recall"], 4)
    res["update_throughput_ops_s"] = round(res["update_throughput_ops_s"], 1)
    _append_rows(_csv_path("exp3", "ablation_final.csv"), [res])
    hist = []
    for h in out["history"]:
        h["policy"] = policy
        h["recall"] = round(h["recall"], 4)
        hist.append(h)
    _append_rows(_csv_path("exp3", "ablation_history.csv"), hist)
    return f"exp3:{policy}"


# ======================================================================= #
# EXP4 -- scalability                                                      #
# ======================================================================= #
EXP4_N = [2000, 5000, 10000, 15000, 20000]
EXP4_D = [32, 64, 128, 256]


def exp4_pending():
    done = _done_units(_csv_path("exp4", "scalability.csv"), "unit_id")
    units = [{"axis": "N", "N": n, "D": 128, "unit_id": f"N{n}"} for n in EXP4_N]
    units += [{"axis": "D", "N": 10000, "D": d, "unit_id": f"D{d}"} for d in EXP4_D]
    return [u for u in units if u["unit_id"] not in done]


def exp4_run_one(unit):
    N, D = unit["N"], unit["D"]
    n_ops = max(3000, N // 3)
    cfg = ExperimentConfig(name="exp4", dataset="synthetic", n_base=N, dim=D, n_query=300,
                           metric="l2", M=16, ef_construction=100, k=10, ef_search=64,
                           n_ops=n_ops, repair_budget=64, repair_interval=256, seed=42)
    base, queries = make_synthetic(n_base=N, dim=D, n_query=cfg.n_query, seed=cfg.seed)
    sids = seed_ids(base, 0.5)
    ad = SystemAdapter("streamvec", cfg)
    t_build = time.perf_counter()
    for i in sids:
        ad.apply("insert", int(i), base[i])
    build_s = time.perf_counter() - t_build  # cost of a from-scratch build of |sids| nodes
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, seed=cfg.seed)
    ins, dele, upd, rep = [], [], [], []
    for n, (op, vid, vec) in enumerate(ops, 1):
        t0 = time.perf_counter()
        ad.apply(op, vid, vec)
        dt = (time.perf_counter() - t0) * 1000
        (ins if op == "insert" else dele if op == "delete" else upd).append(dt)
        if n % 256 == 0:
            t0 = time.perf_counter()
            ad.sys.repair(budget=64)
            rep.append((time.perf_counter() - t0) * 1000)
    ad.finalize()
    r, pc = measure_recall(ad, queries, cfg.k, cfg.ef_search)
    mean = lambda a: round(float(np.mean(a)), 4) if a else 0.0
    n_seed = len(sids)
    row = {"unit_id": unit["unit_id"], "axis": unit["axis"], "N": N, "D": D,
           "n_seed": n_seed, "build_s": round(build_s, 4),
           "build_ms_per_node": round(build_s / max(n_seed, 1) * 1000, 4),
           "insert_ms": mean(ins), "delete_ms": mean(dele), "update_ms": mean(upd),
           "repair_ms_per_call": mean(rep), "query_p50_ms": round(pc["p50_ms"], 4),
           "query_p99_ms": round(pc["p99_ms"], 4), "recall_at_10": round(r, 4),
           "memory_mb": round(estimate_index_memory_mb(ad.sys), 3),
           "dist_calls": ad.sys.graph.dist_calls}
    _append_rows(_csv_path("exp4", "scalability.csv"), [row])
    return f"exp4:{unit['unit_id']}"


# ======================================================================= #
# EXP5 -- repair budget sensitivity                                        #
# ======================================================================= #
EXP5_BUDGETS = [0, 2, 4, 8, 16, 32, -1]


def _exp5_cfg():
    return ExperimentConfig(name="exp5", dataset="sift", n_base=8000, n_query=400, metric="l2",
                            M=8, ef_construction=80, k=10, ef_search=20, n_ops=7000,
                            insert_ratio=0.35, delete_ratio=0.5, update_ratio=0.15,
                            repair_interval=128, ef_repair=48, tombstone_threshold=0.99, seed=42)


def exp5_pending():
    done = _done_units(_csv_path("exp5", "repair_budget.csv"), "repair_budget")
    out = []
    for b in EXP5_BUDGETS:
        label = "all" if b == -1 else str(b)
        if label not in done:
            out.append({"budget": b})
    return out


def exp5_run_one(unit):
    cfg = _exp5_cfg()
    cfg.repair_budget = unit["budget"]
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, 0.5)
    seed_inserts = [(i, base[i]) for i in sids]
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=0.5, seed=cfg.seed)
    ad = SystemAdapter("streamvec", cfg)
    out = run_stream(ad, seed_inserts, ops, queries, cfg, eval_ef=cfg.ef_search, checkpoints=False)
    res = out["result"]
    row = {"repair_budget": "all" if unit["budget"] == -1 else str(unit["budget"]),
           "final_recall": round(res["final_recall"], 4),
           "update_throughput_ops_s": round(res["update_throughput_ops_s"], 1),
           "repair_count": res.get("repair_count", 0),
           "repair_dist_calls": ad.sys.repair_engine.repair_dist_calls,
           "total_dist_calls": res.get("dist_calls", 0),
           "tombstone_ratio": round(res.get("tombstone_ratio", 0.0), 4),
           "memory_mb": round(res["memory_mb"], 2)}
    _append_rows(_csv_path("exp5", "repair_budget.csv"), [row])
    return f"exp5:budget={unit['budget']}"


# ======================================================================= #
# EXP6 -- RAG document-store churn case study                              #
# ======================================================================= #
def _exp6_cfg():
    return ExperimentConfig(name="exp6", dataset="sift", n_base=12000, n_query=400, metric="l2",
                            M=8, ef_construction=80, k=10, ef_search=25,
                            repair_budget=16, repair_interval=128, ef_repair=48,
                            rebuild_interval=4000, tombstone_threshold=0.25, seed=7)


def _exp6_ops(base, cfg):
    rng = np.random.default_rng(cfg.seed)
    window = cfg.n_base // 2
    n_steps = 6000
    ops = []
    live = list(range(window))
    nxt = window
    for _ in range(n_steps):
        r = rng.random()
        if r < 0.2 and len(live) > 100:
            vid = int(live[int(rng.integers(len(live)))])
            ops.append(("update", vid, base[int(rng.integers(len(base)))]))
        else:
            ops.append(("insert", nxt, base[nxt % len(base)]))
            live.append(nxt); nxt += 1
            if len(live) > window:
                ops.append(("delete", live.pop(0), None))
    return ops, window


def exp6_pending():
    done = _done_units(_csv_path("exp6", "rag_quality.csv"), "kind")
    return [{"kind": k} for k in ("streamvec", "hnsw_rebuild") if k not in done]


def exp6_run_one(unit):
    cfg = _exp6_cfg()
    kind = unit["kind"]
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    ops, window = _exp6_ops(base, cfg)
    ad = SystemAdapter(kind, cfg)
    for i in range(window):
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    rows = []
    for n, (op, vid, vec) in enumerate(ops, 1):
        ad.apply(op, vid, vec)
        ad.maintain(n)
        if n % 2000 == 0:
            if ad.is_sv:
                ad.sys.repair(budget=-1)
            else:
                ad.finalize()
            mat, ids = ad.live_vectors()
            gt = compute_groundtruth(mat, ids, queries, k=cfg.k)
            preds = [[i for _, i in ad.search(q, k=cfg.k, ef=cfg.ef_search)] for q in queries]
            rows.append({"kind": kind, "system": ad.name, "step": n,
                         "recall_at_10": round(recall_at_k(preds, gt, cfg.k), 4),
                         "ndcg_at_10": round(ndcg_at_k(preds, gt, cfg.k), 4)})
    _append_rows(_csv_path("exp6", "rag_quality.csv"), rows)
    return f"exp6:{ad.name}"


# ======================================================================= #
# EXP7 -- graph-degree (M) sensitivity of the repair benefit               #
# ======================================================================= #
EXP7_M = [6, 8, 12, 16, 24]


def _exp7_cfg(M):
    return ExperimentConfig(name="exp7", dataset="sift", n_base=7000, n_query=400, metric="l2",
                            M=M, ef_construction=80, k=10, ef_search=20, n_ops=6000,
                            insert_ratio=0.35, delete_ratio=0.5, update_ratio=0.15,
                            repair_budget=16, repair_interval=128, ef_repair=48,
                            tombstone_threshold=0.99, seed=42)


def exp7_pending():
    done = _done_units(_csv_path("exp7", "degree_sensitivity.csv"), "unit_id")
    units = []
    for M in EXP7_M:
        for variant in ("repair", "norepair"):
            uid = f"M{M}_{variant}"
            if uid not in done:
                units.append({"M": M, "variant": variant, "unit_id": uid})
    return units


def exp7_run_one(unit):
    M, variant = unit["M"], unit["variant"]
    cfg = _exp7_cfg(M)
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, cfg.seed_fraction)
    seed_inserts = [(i, base[i]) for i in sids]
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=cfg.seed_fraction, seed=cfg.seed)
    kind = "streamvec" if variant == "repair" else "streamvec_norepair"
    ad = SystemAdapter(kind, cfg)
    out = run_stream(ad, seed_inserts, ops, queries, cfg, eval_ef=cfg.ef_search, checkpoints=False)
    res = out["result"]
    row = {"unit_id": unit["unit_id"], "M": M, "variant": variant, "system": ad.name,
           "final_recall": round(res["final_recall"], 4),
           "query_qps": round(res.get("query_qps", 0.0), 1),
           "update_throughput_ops_s": round(res["update_throughput_ops_s"], 1),
           "tombstone_ratio": round(res.get("tombstone_ratio", 0.0), 4)}
    _append_rows(_csv_path("exp7", "degree_sensitivity.csv"), [row])
    print(f"  exp7 M={M} {variant} recall={row['final_recall']}", flush=True)
    return f"exp7:{unit['unit_id']}"


# ======================================================================= #
# EXP8 -- freshness / staleness vs. rebuild interval                       #
# ======================================================================= #
# StreamVec applies every op immediately (always fresh); a periodic-rebuild
# index reflects updates only at rebuild points, so between rebuilds it answers
# from a STALE snapshot.  We sample Recall@10 frequently against the CURRENT live
# set, so the rebuild baseline's stale windows lower its time-averaged recall.
EXP8_R = [1000, 2000, 4000, 8000]


def _exp8_cfg():
    return ExperimentConfig(name="exp8", dataset="sift", n_base=10000, n_query=300, metric="l2",
                            M=16, ef_construction=100, k=10, ef_search=40, n_ops=12000,
                            insert_ratio=0.5, delete_ratio=0.25, update_ratio=0.25,
                            repair_budget=64, repair_interval=256, ef_repair=64,
                            tombstone_threshold=0.99, seed=11)


def exp8_pending():
    done = _done_units(_csv_path("exp8", "freshness_summary.csv"), "unit_id")
    # rebuild baseline uses the optimized C++ HNSW so the comparison is
    # conservative: the baseline has fresh-quality recall whenever it rebuilds,
    # and only *staleness between rebuilds* lowers its time-averaged recall.
    # We also include the same-runtime tombstone baselines (NoRepair, Append):
    # these apply every op immediately so they are *also* fresh -- showing that
    # the freshness advantage is specifically over periodic rebuild, while the
    # advantage over tombstone baselines is recall (NoRepair) / query cost (Append).
    units = [{"unit_id": "streamvec", "kind": "streamvec", "R": 0}]
    units += [{"unit_id": f"rebuild_R{R}", "kind": "hnsw_rebuild", "R": R} for R in EXP8_R]
    units += [{"unit_id": "norepair", "kind": "streamvec_norepair", "R": 0},
              {"unit_id": "append", "kind": "hnsw_append", "R": 0}]
    return [u for u in units if u["unit_id"] not in done]


def exp8_run_one(unit):
    cfg = _exp8_cfg()
    if unit["R"]:
        cfg.rebuild_interval = unit["R"]
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, 0.5)
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=0.5, seed=cfg.seed)
    ad = SystemAdapter(unit["kind"], cfg)
    for i in sids:
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    sample_every = 500
    t0 = time.perf_counter()
    series = []
    for n, (op, vid, vec) in enumerate(ops, 1):
        ad.apply(op, vid, vec)
        ad.maintain(n)
        if n % sample_every == 0:
            mat, ids = ad.live_vectors()          # CURRENT live set (truth)
            gt = compute_groundtruth(mat, ids, queries, k=cfg.k)
            preds = [[i for _, i in ad.search(q, k=cfg.k, ef=cfg.ef_search)] for q in queries]
            r = recall_at_k(preds, gt, cfg.k)      # stale search vs current truth
            series.append(r)
            _append_rows(_csv_path("exp8", "freshness_series.csv"),
                         [{"unit_id": unit["unit_id"], "system": ad.name, "R": unit["R"],
                           "ops": n, "recall_at_10": round(r, 4)}])
    wall = time.perf_counter() - t0
    rebuilds = getattr(ad.sys, "rebuild_count", 0) if not ad.is_sv else 0
    summary = {"unit_id": unit["unit_id"], "system": ad.name, "R": unit["R"],
               "time_avg_recall": round(float(np.mean(series)), 4),
               "min_recall": round(float(np.min(series)), 4),
               "amortized_maint_ms_per_op": round(wall / len(ops) * 1000, 4),
               "rebuilds": rebuilds}
    _append_rows(_csv_path("exp8", "freshness_summary.csv"), [summary])
    print(f"  exp8 {unit['unit_id']} time_avg_recall={summary['time_avg_recall']} "
          f"min={summary['min_recall']}", flush=True)
    return f"exp8:{unit['unit_id']}"


# ======================================================================= #
# EXP9 -- workload-mix sensitivity                                          #
# ======================================================================= #
EXP9_MIX = [("insert-heavy", 0.70, 0.15, 0.15),
            ("balanced", 0.34, 0.33, 0.33),
            ("delete-heavy", 0.30, 0.50, 0.20),
            ("update-heavy", 0.30, 0.20, 0.50)]


def exp9_pending():
    done = _done_units(_csv_path("exp9", "workload.csv"), "mix")
    return [{"mix": m, "ins": a, "del": b, "upd": c}
            for (m, a, b, c) in EXP9_MIX if m not in done]


def exp9_run_one(unit):
    cfg = ExperimentConfig(name="exp9", dataset="sift", n_base=8000, n_query=400, metric="l2",
                           M=8, ef_construction=80, k=10, ef_search=25, n_ops=8000,
                           insert_ratio=unit["ins"], delete_ratio=unit["del"], update_ratio=unit["upd"],
                           repair_budget=16, repair_interval=128, ef_repair=48,
                           tombstone_threshold=0.99, seed=42)
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, 0.5)
    seed_inserts = [(i, base[i]) for i in sids]
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=0.5, seed=cfg.seed)
    ad = SystemAdapter("streamvec", cfg)
    out = run_stream(ad, seed_inserts, ops, queries, cfg, eval_ef=cfg.ef_search, checkpoints=False)
    res = out["result"]
    row = {"mix": unit["mix"], "insert_ratio": unit["ins"], "delete_ratio": unit["del"],
           "update_ratio": unit["upd"], "final_recall": round(res["final_recall"], 4),
           "query_qps": round(res.get("query_qps", 0), 1),
           "update_throughput_ops_s": round(res["update_throughput_ops_s"], 1),
           "tombstone_ratio": round(res.get("tombstone_ratio", 0.0), 4),
           "repair_count": res.get("repair_count", 0)}
    _append_rows(_csv_path("exp9", "workload.csv"), [row])
    print(f"  exp9 {unit['mix']} recall={row['final_recall']} tomb={row['tombstone_ratio']}", flush=True)
    return f"exp9:{unit['mix']}"


# ======================================================================= #
# EXP10 -- adaptive rebuild trigger dynamics                               #
# ======================================================================= #
def exp10_pending():
    done = _done_units(_csv_path("exp10", "trigger_series.csv"), "mode")
    return [{"mode": m} for m in ("trigger_on", "trigger_off") if m not in done]


def exp10_run_one(unit):
    mode = unit["mode"]
    tau = 0.15 if mode == "trigger_on" else 0.99
    cfg = ExperimentConfig(name="exp10", dataset="sift", n_base=6000, n_query=300, metric="l2",
                           M=12, ef_construction=80, k=10, ef_search=40, n_ops=8000,
                           insert_ratio=0.4, delete_ratio=0.4, update_ratio=0.2,
                           repair_budget=48, repair_interval=200, ef_repair=56,
                           tombstone_threshold=tau, seed=5)
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, 0.5)
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=0.5, seed=cfg.seed)
    ad = SystemAdapter("streamvec", cfg)
    for i in sids:
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    rows = []
    for n, (op, vid, vec) in enumerate(ops, 1):
        ad.apply(op, vid, vec)
        ad.maintain(n)
        if n % 1000 == 0:
            st = ad.sys.get_stats()
            r, pc = measure_recall(ad, queries, cfg.k, cfg.ef_search)
            rows.append({"mode": mode, "ops": n,
                         "tombstone_ratio": round(st.tombstone_ratio, 4),
                         "memory_mb": round(estimate_index_memory_mb(ad.sys), 3),
                         "query_p50_ms": round(pc["p50_ms"], 4),
                         "recall_at_10": round(r, 4),
                         "rebuilds": ad.rebuilds})
            _append_rows(_csv_path("exp10", "trigger_series.csv"), rows[-1:])
    print(f"  exp10 {mode} final_tomb={rows[-1]['tombstone_ratio']} rebuilds={rows[-1]['rebuilds']}", flush=True)
    return f"exp10:{mode}"


# ======================================================================= #
# EXP11 -- generality on a second dataset (GloVe-100, cosine)              #
# ======================================================================= #
EXP11_TOMB = [0.0, 0.1, 0.2, 0.3, 0.4]


def _exp11_cfg():
    return ExperimentConfig(name="exp11", dataset="glove", n_base=8000, n_query=400, metric="cosine",
                            M=8, ef_construction=80, k=10, ef_search=25,
                            repair_policy="priority", repair_budget=16, ef_repair=48, seed=23)


def exp11_pending():
    done = _done_units(_csv_path("exp11", "glove_recall_vs_tombstone.csv"), "kind")
    return [{"kind": k} for k in ("streamvec", "streamvec_norepair", "hnsw_append") if k not in done]


def exp11_run_one(unit):
    cfg = _exp11_cfg()
    kind = unit["kind"]
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    rng = np.random.default_rng(cfg.seed)
    delete_order = rng.permutation(cfg.n_base)
    ad = SystemAdapter(kind, cfg)
    for i in range(cfg.n_base):
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    deleted = 0
    rows = []
    for lvl in EXP11_TOMB:
        target = int(lvl * cfg.n_base)
        while deleted < target:
            ad.apply("delete", int(delete_order[deleted]), None)
            deleted += 1
            if kind == "streamvec" and deleted % 256 == 0:
                ad.sys.repair(budget=cfg.repair_budget)
        if kind == "streamvec":
            ad.sys.repair(budget=cfg.repair_budget)
        r, pc = measure_recall(ad, queries, cfg.k, cfg.ef_search)
        rows.append({"kind": kind, "system": ad.name, "tombstone_ratio": lvl,
                     "recall_at_10": round(r, 4), "query_qps": round(pc["qps"], 1)})
        print(f"  exp11 {kind} tomb={lvl:.1f} recall={r:.3f}", flush=True)
    _append_rows(_csv_path("exp11", "glove_recall_vs_tombstone.csv"), rows)
    return f"exp11:{ad.name}"


# ======================================================================= #
# EXP12 -- control-parameter sensitivity (tau_th, Delta_rep, P, ef_r)       #
# ======================================================================= #
EXP12_SWEEPS = {
    "tau_th":    [0.10, 0.15, 0.25, 0.50, 0.99],
    "delta_rep": [64, 128, 256, 512],
    "P":         [0.25, 0.5, 0.75, 1.0],
    "ef_r":      [16, 32, 48, 64, 96],
}


def _exp12_cfg():
    return ExperimentConfig(name="exp12", dataset="sift", n_base=6000, n_query=400, metric="l2",
                            M=8, ef_construction=80, k=10, ef_search=20, n_ops=5000,
                            insert_ratio=0.35, delete_ratio=0.5, update_ratio=0.15,
                            repair_budget=16, repair_interval=128, ef_repair=48,
                            tombstone_threshold=0.15, degree_deficit_threshold=0.3, seed=42)


def exp12_pending():
    done = _done_units(_csv_path("exp12", "param_sensitivity.csv"), "unit_id")
    units = []
    for param, vals in EXP12_SWEEPS.items():
        for v in vals:
            uid = f"{param}={v}"
            if uid not in done:
                units.append({"param": param, "value": v, "unit_id": uid})
    return units


def exp12_run_one(unit):
    cfg = _exp12_cfg()
    param, value = unit["param"], unit["value"]
    # tau_th sweep enables the rebuild trigger; other sweeps disable it to
    # isolate the parameter under test.
    if param == "tau_th":
        cfg.tombstone_threshold = value
    else:
        cfg.tombstone_threshold = 0.99
        if param == "delta_rep":
            cfg.repair_interval = value
        elif param == "ef_r":
            cfg.ef_repair = value
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    sids = seed_ids(base, 0.5)
    seed_inserts = [(i, base[i]) for i in sids]
    ops, _ = generate_stream(base, n_ops=cfg.n_ops, insert_ratio=cfg.insert_ratio,
                             delete_ratio=cfg.delete_ratio, update_ratio=cfg.update_ratio,
                             seed_fraction=0.5, seed=cfg.seed)
    ad = SystemAdapter("streamvec", cfg)
    if param == "P":
        ad.sys.tombstone_prune_fraction = value
    out = run_stream(ad, seed_inserts, ops, queries, cfg, eval_ef=cfg.ef_search, checkpoints=False)
    res = out["result"]
    row = {"unit_id": unit["unit_id"], "param": param, "value": value,
           "final_recall": round(res["final_recall"], 4),
           "query_qps": round(res.get("query_qps", 0.0), 1),
           "update_throughput_ops_s": round(res["update_throughput_ops_s"], 1),
           "rebuilds": res.get("rebuilds", 0),
           "repair_count": res.get("repair_count", 0),
           "repair_dist_calls": ad.sys.repair_engine.repair_dist_calls,
           "tombstone_ratio": round(res.get("tombstone_ratio", 0.0), 4),
           "memory_mb": round(res["memory_mb"], 2)}
    _append_rows(_csv_path("exp12", "param_sensitivity.csv"), [row])
    print(f"  exp12 {unit['unit_id']} recall={row['final_recall']} rebuilds={row['rebuilds']}", flush=True)
    return f"exp12:{unit['unit_id']}"


# ======================================================================= #
# EXP13 -- adversarial deletions + query-driven (hotness) scheduling        #
# ======================================================================= #
def exp13_pending():
    adv_done = _done_units(_csv_path("exp13", "adversarial.csv"), "unit_id")
    hot_done = _done_units(_csv_path("exp13", "hotness.csv"), "unit_id")
    units = []
    for target in ("hub", "random"):
        for variant in ("repair", "norepair"):
            uid = f"adv_{target}_{variant}"
            if uid not in adv_done:
                units.append({"study": "adv", "target": target, "variant": variant, "unit_id": uid})
    for policy in ("priority", "hotness", "norepair"):
        uid = f"hot_{policy}"
        if uid not in hot_done:
            units.append({"study": "hot", "policy": policy, "unit_id": uid})
    return units


def _exp13_adv(unit):
    target, variant = unit["target"], unit["variant"]
    cfg = ExperimentConfig(name="exp13", dataset="sift", n_base=7000, n_query=400, metric="l2",
                           M=8, ef_construction=80, k=10, ef_search=20,
                           repair_budget=16, ef_repair=48, tombstone_threshold=0.99, seed=42)
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    kind = "streamvec" if variant == "repair" else "streamvec_norepair"
    ad = SystemAdapter(kind, cfg)
    for i in range(cfg.n_base):
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    # deletion order: hub = highest current live degree first; random = shuffled
    if target == "hub":
        deg = {nid: len(ad.sys.graph.neighbors[nid][0]) for nid in ad.sys.graph.id_to_row}
        order = [nid for nid, _ in sorted(deg.items(), key=lambda kv: -kv[1])]
    else:
        rng = np.random.default_rng(cfg.seed)
        order = list(ad.sys.graph.id_to_row.keys())
        rng.shuffle(order)
    rows = []
    deleted = 0
    for lvl in (0.0, 0.1, 0.2, 0.3, 0.4):
        tgt = int(lvl * cfg.n_base)
        while deleted < tgt and deleted < len(order):
            ad.apply("delete", int(order[deleted]), None)
            deleted += 1
            if variant == "repair" and deleted % 256 == 0:
                ad.sys.repair(budget=cfg.repair_budget)
        if variant == "repair":
            ad.sys.repair(budget=cfg.repair_budget)
        r, _ = measure_recall(ad, queries, cfg.k, cfg.ef_search)
        rows.append({"unit_id": unit["unit_id"], "target": target, "variant": variant,
                     "tombstone_ratio": lvl, "recall_at_10": round(r, 4)})
        print(f"  exp13 adv {target}/{variant} tomb={lvl:.1f} recall={r:.3f}", flush=True)
    _append_rows(_csv_path("exp13", "adversarial.csv"), rows)
    return f"exp13:{unit['unit_id']}"


def _exp13_hot(unit):
    # Skewed query workload: queries concentrate on a "hot" region; deletions
    # damage nodes across the index.  Hotness-aware repair should protect the
    # hot region better than degree-deficit alone under a tight budget.
    policy = unit["policy"]
    cfg = ExperimentConfig(name="exp13", dataset="sift", n_base=7000, n_query=600, metric="l2",
                           M=8, ef_construction=80, k=10, ef_search=20,
                           repair_budget=8, ef_repair=48, tombstone_threshold=0.99, seed=3)
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    rng = np.random.default_rng(cfg.seed)
    # hot queries: a fixed small set repeated; evaluation uses the same hot set
    hot_q = queries[:80]
    kind = "streamvec_norepair" if policy == "norepair" else "streamvec"
    ad = SystemAdapter(kind, cfg, repair_policy=(None if policy == "norepair" else policy))
    for i in range(cfg.n_base):
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    order = list(ad.sys.graph.id_to_row.keys())
    rng.shuffle(order)
    deleted = 0
    n_rounds = 24
    dels_per_round = int(0.5 * cfg.n_base / n_rounds)
    for _ in range(n_rounds):
        for _ in range(dels_per_round):
            if deleted < len(order):
                ad.apply("delete", int(order[deleted]), None); deleted += 1
        # issue hot queries to accumulate visit frequency (drives hotness key)
        for q in hot_q:
            ad.sys.search(q, k=cfg.k, ef_search=cfg.ef_search)
        if policy != "norepair":
            ad.sys.repair(budget=cfg.repair_budget)
    # evaluate recall on the hot set and on a uniform set
    def rec(qset):
        mat, ids = ad.live_vectors()
        gt = compute_groundtruth(mat, ids, qset, k=cfg.k)
        preds = [[i for _, i in ad.sys.search(q, k=cfg.k, ef_search=cfg.ef_search)] for q in qset]
        return round(recall_at_k(preds, gt, cfg.k), 4)
    hot_r = rec(hot_q)
    uni_r = rec(queries[80:380])
    row = {"unit_id": unit["unit_id"], "policy": policy, "hot_recall": hot_r, "uniform_recall": uni_r}
    _append_rows(_csv_path("exp13", "hotness.csv"), [row])
    print(f"  exp13 hot {policy} hot_recall={hot_r} uniform_recall={uni_r}", flush=True)
    return f"exp13:{unit['unit_id']}"


def exp13_run_one(unit):
    return _exp13_adv(unit) if unit["study"] == "adv" else _exp13_hot(unit)


# ======================================================================= #
# EXP14 -- per-query operation microbenchmark vs tombstone ratio            #
# ======================================================================= #
# Empirically bounds the constant factors Proposition 1 omits: distance
# evaluations, adjacency-entry scans (incl. dead slots), and heap operations
# per query as the tombstone ratio grows (no repair, to isolate skip overhead).
def exp14_pending():
    done = _done_units(_csv_path("exp14", "microbench.csv"), "tombstone_ratio")
    return [{"all": True}] if not done else []


def exp14_run_one(unit):
    cfg = ExperimentConfig(name="exp14", dataset="sift", n_base=10000, n_query=300, metric="l2",
                           M=16, ef_construction=100, k=10, ef_search=64, seed=42)
    base, queries = load_dataset(cfg.dataset, n_base=cfg.n_base, n_query=cfg.n_query, seed=cfg.seed)
    cfg.dim = base.shape[1]
    ad = SystemAdapter("streamvec_norepair", cfg)   # no repair: pure skip overhead
    for i in range(cfg.n_base):
        ad.apply("insert", int(i), base[i])
    ad.finalize()
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(cfg.n_base)
    deleted = 0
    rows = []
    for lvl in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        tgt = int(lvl * cfg.n_base)
        while deleted < tgt:
            ad.apply("delete", int(order[deleted]), None); deleted += 1
        g = ad.sys.graph
        g.dist_calls = g.neighbor_checks = g.heap_ops = 0
        nq = len(queries)
        for q in queries:
            ad.sys.search(q, k=cfg.k, ef_search=cfg.ef_search)
        rows.append({"tombstone_ratio": lvl,
                     "dist_per_query": round(g.dist_calls / nq, 1),
                     "nchecks_per_query": round(g.neighbor_checks / nq, 1),
                     "heapops_per_query": round(g.heap_ops / nq, 1)})
        print(f"  exp14 tomb={lvl:.1f} dist={rows[-1]['dist_per_query']} "
              f"nchecks={rows[-1]['nchecks_per_query']}", flush=True)
    _append_rows(_csv_path("exp14", "microbench.csv"), rows)
    return "exp14:done"


# ======================================================================= #
REGISTRY = {
    "exp1": (exp1_pending, exp1_run_one),
    "exp2": (exp2_pending, exp2_run_one),
    "exp3": (exp3_pending, exp3_run_one),
    "exp4": (exp4_pending, exp4_run_one),
    "exp5": (exp5_pending, exp5_run_one),
    "exp6": (exp6_pending, exp6_run_one),
    "exp7": (exp7_pending, exp7_run_one),
    "exp8": (exp8_pending, exp8_run_one),
    "exp9": (exp9_pending, exp9_run_one),
    "exp10": (exp10_pending, exp10_run_one),
    "exp11": (exp11_pending, exp11_run_one),
    "exp12": (exp12_pending, exp12_run_one),
    "exp13": (exp13_pending, exp13_run_one),
    "exp14": (exp14_pending, exp14_run_one),
}
