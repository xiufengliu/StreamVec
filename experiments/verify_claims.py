"""Phase-5 audit: cross-check every numeric claim in the paper against the
result CSVs.  Prints PASS/FAIL per claim.  No result is hand-entered here that
is not derived from results/*.csv -- this script IS the anti-fabrication check.
"""
import csv, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def load(exp, name):
    with open(os.path.join(RES, exp, name)) as f:
        return list(csv.DictReader(f))


def approx(a, b, tol):
    return abs(a - b) <= tol


checks = []


def check(desc, cond, detail=""):
    checks.append((desc, cond, detail))
    print(("PASS" if cond else "FAIL"), "-", desc, ("| " + detail) if detail else "")


# ---- Exp1 ----
p = load("exp1", "pareto.csv")
sv = [r for r in p if r["system"] == "StreamVec"]
sv_max = max(float(r["recall_at_10"]) for r in sv)
check("Exp1: StreamVec peak Recall@10 ~0.99 (paper: up to 0.992)",
      approx(sv_max, 0.992, 0.01), f"measured {sv_max:.4f}")
for sysname in ("Python-Rebuild", "HNSW-Append", "HNSW-Rebuild", "FAISS-IVF-Rebuild"):
    assert any(r["system"] == sysname for r in p), f"missing {sysname}"
check("Exp1: all 5 systems present (StreamVec + 4 baselines)", True)

# ---- Exp2 ----
e2 = load("exp2", "recall_vs_tombstone.csv")
ap = [r for r in e2 if r["kind"] == "hnsw_append"]
ap0 = [float(r["query_qps"]) for r in ap if float(r["tombstone_ratio"]) == 0][0]
ap3 = [float(r["query_qps"]) for r in ap if float(r["tombstone_ratio"]) == 0.3][0]
drop = (1 - ap3 / ap0) * 100
check("Exp2: Append QPS drop at tau=0.3 ~18% (paper claim)",
      approx(drop, 18.0, 3.0), f"measured {drop:.1f}%")
svr = [r for r in e2 if r["kind"] == "streamvec"]
sv_q0 = [float(r["query_qps"]) for r in svr if float(r["tombstone_ratio"]) == 0][0]
sv_q3 = [float(r["query_qps"]) for r in svr if float(r["tombstone_ratio"]) == 0.3][0]
sv_drop = abs(1 - sv_q3 / sv_q0) * 100
check("Exp2: StreamVec QPS ~invariant to churn (<6% change)",
      sv_drop < 6.0, f"change {sv_drop:.1f}%")
sv_r0 = [float(r["recall_at_10"]) for r in svr if float(r["tombstone_ratio"]) == 0][0]
sv_r4 = [float(r["recall_at_10"]) for r in svr if float(r["tombstone_ratio"]) == 0.4][0]
check("Exp2: StreamVec recall stays within ~3% over churn",
      (sv_r0 - sv_r4) <= 0.035, f"{sv_r0:.3f}->{sv_r4:.3f}")

# ---- Exp3 ----
e3 = {r["policy"]: r for r in load("exp3", "ablation_final.csv")}
pri, eag, nor = e3["priority"], e3["eager"], e3["norepair"]
tput_ratio = float(pri["update_throughput_ops_s"]) / float(eag["update_throughput_ops_s"])
dist_ratio = float(eag["repair_dist_calls"]) / float(pri["repair_dist_calls"])
check("Exp3: Priority vs Eager throughput ~3.4x (paper)",
      approx(tput_ratio, 3.4, 0.5), f"{tput_ratio:.2f}x")
check("Exp3: Eager vs Priority repair dist-calls ~5.5x (paper)",
      approx(dist_ratio, 5.5, 1.0), f"{dist_ratio:.2f}x")
gap = float(pri["final_recall"]) - float(nor["final_recall"])
check("Exp3: repair recovers ~3% recall over NoRepair",
      approx(gap, 0.03, 0.015), f"gap {gap:.3f}")
check("Exp3: Priority matches Eager recall (within 1%)",
      abs(float(pri["final_recall"]) - float(eag["final_recall"])) < 0.01,
      f"{pri['final_recall']} vs {eag['final_recall']}")

# ---- Exp4 ----
e4 = load("exp4", "scalability.csv")
N = sorted([r for r in e4 if r["axis"] == "N"], key=lambda r: int(r["N"]))
ins = [float(r["insert_ms"]) for r in N]
bld = [float(r["build_s"]) for r in N]
mem = [float(r["memory_mb"]) for r in N]
check("Exp4: insert latency ~flat in N (max/min < 1.4x => sub-linear)",
      max(ins) / min(ins) < 1.4, f"{min(ins):.2f}->{max(ins):.2f} ms")
check("Exp4: full-rebuild time grows ~linearly with N",
      bld[-1] / bld[0] > 5, f"{bld[0]:.2f}->{bld[-1]:.2f} s over 10x N")
check("Exp4: memory grows ~linearly with N",
      mem[-1] / mem[0] > 3.5, f"{mem[0]:.1f}->{mem[-1]:.1f} MB")

# ---- Exp5 ----
e5 = {r["repair_budget"]: r for r in load("exp5", "repair_budget.csv")}
allr = float(e5["all"]["final_recall"])
smallr = float(e5["0"]["final_recall"])
check("Exp5: recall ~flat across budgets (|all-0| < 1%)",
      abs(allr - smallr) < 0.01, f"budget0={smallr:.4f} all={allr:.4f}")
cost_ratio = float(e5["all"]["repair_dist_calls"]) / float(e5["8"]["repair_dist_calls"])
check("Exp5: eager(all) repair cost >> small budget (>3x dist calls)",
      cost_ratio > 3, f"{cost_ratio:.1f}x")

# ---- Exp6 ----
e6 = load("exp6", "rag_quality.csv")
sv6 = [float(r["recall_at_10"]) for r in e6 if r["kind"] == "streamvec"]
rb6 = [float(r["recall_at_10"]) for r in e6 if r["kind"] == "hnsw_rebuild"]
check("Exp6: StreamVec recall range 0.96-0.97 (paper)",
      min(sv6) >= 0.96 and max(sv6) <= 0.975, f"{min(sv6):.3f}-{max(sv6):.3f}")
check("Exp6: StreamVec >= periodic Rebuild on recall",
      min(sv6) >= max(rb6) - 0.01, f"sv_min={min(sv6):.3f} rb_max={max(rb6):.3f}")

# ---- Exp7 (degree sensitivity) ----
e7 = {(r["M"], r["variant"]): float(r["final_recall"])
      for r in load("exp7", "degree_sensitivity.csv")}
gain6 = e7[("6", "repair")] - e7[("6", "norepair")]
gain16 = e7[("16", "repair")] - e7[("16", "norepair")]
check("Exp7: repair gain ~4 pts at M=6 (paper: +4.2)",
      approx(gain6 * 100, 4.2, 0.6), f"+{gain6*100:.1f} pts")
check("Exp7: repair gain ~1 pt at M=16 (paper: +1.1)",
      approx(gain16 * 100, 1.1, 0.6), f"+{gain16*100:.1f} pts")
check("Exp7: gain decreases monotonically with M",
      gain6 > (e7[("8","repair")]-e7[("8","norepair")]) >
      (e7[("12","repair")]-e7[("12","norepair")]) > gain16, "")

# ---- Exp8 (freshness) ----
e8 = {r["unit_id"]: r for r in load("exp8", "freshness_summary.csv")}
sv_tavg = float(e8["streamvec"]["time_avg_recall"])
check("Exp8: StreamVec time-avg recall ~0.987 (paper)",
      approx(sv_tavg, 0.987, 0.005), f"{sv_tavg:.4f}")
rb1k = float(e8["rebuild_R1000"]["time_avg_recall"])
rb8k = float(e8["rebuild_R8000"]["time_avg_recall"])
check("Exp8: rebuild R=1000 ~0.95, R=8000 ~0.69 (paper)",
      approx(rb1k, 0.952, 0.01) and approx(rb8k, 0.686, 0.02), f"R1k={rb1k:.3f} R8k={rb8k:.3f}")
check("Exp8: StreamVec > best rebuild (R=1000)", sv_tavg > rb1k, f"{sv_tavg:.3f}>{rb1k:.3f}")
check("Exp8: rebuild min dips to ~0.41 at R=8000 (paper)",
      approx(float(e8["rebuild_R8000"]["min_recall"]), 0.41, 0.05),
      f"{e8['rebuild_R8000']['min_recall']}")

# ---- Exp9 (workload) ----
e9 = {r["mix"]: r for r in load("exp9", "workload.csv")}
recs = [float(e9[m]["final_recall"]) for m in e9]
check("Exp9: recall in [0.94,0.96] across all mixes (paper)",
      min(recs) >= 0.94 and max(recs) <= 0.965, f"{min(recs):.3f}-{max(recs):.3f}")

# ---- Exp10 (adaptive trigger) ----
series = load("exp10", "trigger_series.csv")
def last_mem(mode):
    rs = [r for r in series if r["mode"] == mode]
    return float(rs[-1]["memory_mb"]), max(float(r["tombstone_ratio"]) for r in rs)
mem_on, tomb_on = last_mem("trigger_on")
mem_off, tomb_off = last_mem("trigger_off")
check("Exp10: trigger bounds memory (~3.7 vs ~9.3 MB, paper)",
      approx(mem_on, 3.7, 0.4) and approx(mem_off, 9.3, 0.6), f"on={mem_on:.1f} off={mem_off:.1f}")
check("Exp10: trigger bounds tombstone ratio (off grows >0.4)",
      tomb_off > 0.4 and tomb_on < 0.2, f"on={tomb_on:.2f} off={tomb_off:.2f}")

# ---- Exp11 (GloVe generality) ----
e11 = load("exp11", "glove_recall_vs_tombstone.csv")
ap = [r for r in e11 if r["system"] == "HNSW-Append"]
ap0 = [float(r["query_qps"]) for r in ap if float(r["tombstone_ratio"]) == 0][0]
ap3 = [float(r["query_qps"]) for r in ap if float(r["tombstone_ratio"]) == 0.3][0]
check("Exp11: GloVe append QPS drops ~22% at tau=0.3 (paper)",
      approx((1 - ap3 / ap0) * 100, 22.0, 6.0), f"{(1-ap3/ap0)*100:.1f}%")
sv11 = [r for r in e11 if r["system"] == "StreamVec"]
sv0 = [float(r["query_qps"]) for r in sv11 if float(r["tombstone_ratio"]) == 0][0]
sv3 = [float(r["query_qps"]) for r in sv11 if float(r["tombstone_ratio"]) == 0.3][0]
check("Exp11: GloVe StreamVec QPS ~invariant (<10%)",
      abs(1 - sv3 / sv0) < 0.10, f"{abs(1-sv3/sv0)*100:.1f}%")

# ---- Exp8 augmentation (same-runtime tombstone baselines are fresh) ----
e8b = {r["system"]: float(r["time_avg_recall"]) for r in load("exp8", "freshness_summary.csv")}
check("Exp8: NoRepair & Append are also fresh (time-avg > 0.98)",
      e8b.get("StreamVec-NoRepair", 0) > 0.98 and e8b.get("HNSW-Append", 0) > 0.98,
      f"NoRepair={e8b.get('StreamVec-NoRepair')} Append={e8b.get('HNSW-Append')}")

# ---- Exp12 (parameter sensitivity) ----
e12 = load("exp12", "param_sensitivity.csv")
recs12 = [float(r["final_recall"]) for r in e12]
check("Exp12: recall robust across all params [0.965,0.98]",
      min(recs12) >= 0.965 and max(recs12) <= 0.98, f"{min(recs12):.3f}-{max(recs12):.3f}")
tau = {r["value"]: r for r in e12 if r["param"] == "tau_th"}
check("Exp12: low tau_th rebuilds more (0.1:>=8, 0.99:0)",
      int(tau["0.1"]["rebuilds"]) >= 8 and int(tau["0.99"]["rebuilds"]) == 0,
      f"0.1->{tau['0.1']['rebuilds']} 0.99->{tau['0.99']['rebuilds']}")

# ---- Exp13 (adversarial + hotness) ----
adv = load("exp13", "adversarial.csv")
def adv_rec(tg, var, lvl):
    return [float(r["recall_at_10"]) for r in adv
            if r["target"] == tg and r["variant"] == var and float(r["tombstone_ratio"]) == lvl][0]
check("Exp13: hub deletion <= random at tau=0.1 (more damaging)",
      adv_rec("hub", "repair", 0.1) <= adv_rec("random", "repair", 0.1) + 1e-9,
      f"hub={adv_rec('hub','repair',0.1)} random={adv_rec('random','repair',0.1)}")
hot = {r["policy"]: r for r in load("exp13", "hotness.csv")}
check("Exp13: hotness improves hot-set recall over priority",
      float(hot["hotness"]["hot_recall"]) > float(hot["priority"]["hot_recall"]),
      f"hotness={hot['hotness']['hot_recall']} priority={hot['priority']['hot_recall']}")

# ---- Exp14 (microbenchmark) ----
mb = load("exp14", "microbench.csv")
nchecks = [float(r["nchecks_per_query"]) for r in mb]
dists = [float(r["dist_per_query"]) for r in mb]
check("Exp14: adjacency scans flat vs tombstone ratio (<10% spread)",
      (max(nchecks) - min(nchecks)) / min(nchecks) < 0.10, f"{min(nchecks):.0f}-{max(nchecks):.0f}")
check("Exp14: distance evals decrease with tombstones",
      dists[0] > dists[-1], f"{dists[0]:.0f}->{dists[-1]:.0f}")

# ====================================================================
# C++ implementation results (results_cpp/*.csv, written without headers)
# ====================================================================
import csv as _csv
RESCPP = os.path.join(HERE, "results_cpp")
def loadcpp(name, cols):
    p = os.path.join(RESCPP, name)
    out = []
    if not os.path.exists(p):
        return out
    for r in _csv.reader(open(p)):
        if not r or r[0] == cols[0]:
            continue
        d = {}
        for i, c in enumerate(cols):
            try: d[c] = float(r[i])
            except (ValueError, IndexError): d[c] = r[i]
        out.append(d)
    return out

e2e = loadcpp("e2e.csv", ["system", "N", "ef", "recall", "qps", "utput", "mem"])
if e2e:
    ef40 = {r["system"]: r for r in e2e if r["ef"] == 40}
    sv = ef40.get("StreamVec", {})
    check("C++ e2e: StreamVec recall highest-tier (>=0.985)",
          sv.get("recall", 0) >= 0.985, f"{sv.get('recall')}")
    check("C++ e2e: StreamVec memory <= Append (append bloats)",
          sv.get("mem", 9e9) <= ef40.get("Append", {}).get("mem", 0),
          f"sv={sv.get('mem')} append={ef40.get('Append',{}).get('mem')}")
    check("C++ e2e: ReplaceDeleted lowest recall (stale edges)",
          ef40.get("ReplaceDeleted", {}).get("recall", 1) == min(r["recall"] for r in ef40.values()),
          f"reuse={ef40.get('ReplaceDeleted',{}).get('recall')}")
    check("C++ e2e: all 5 systems present (incl. delta-overlay, slot-reuse)",
          len(ef40) == 5, f"{sorted(ef40)}")

sc = loadcpp("scale.csv", ["N", "bps", "insert_ms", "qp50", "qps", "mem"])
if sc:
    sc.sort(key=lambda r: r["N"])
    check("C++ scale: insert latency sublinear (4x N -> <4x latency)",
          sc[-1]["insert_ms"] / sc[0]["insert_ms"] < 4.0,
          f"{sc[0]['insert_ms']:.2f}->{sc[-1]['insert_ms']:.2f}ms")
    check("C++ scale: memory ~linear in N", sc[-1]["mem"] / sc[0]["mem"] > 3.0,
          f"{sc[0]['mem']:.0f}->{sc[-1]['mem']:.0f}MB")

cc = loadcpp("concur.csv", ["t", "N", "qps", "p50", "p99"])
if cc:
    cc.sort(key=lambda r: r["t"])
    check("C++ concur: QPS scales with threads (3thr > 2.5x 1thr)",
          cc[-1]["qps"] > 2.5 * cc[0]["qps"], f"{cc[0]['qps']:.0f}->{cc[-1]['qps']:.0f}")
    check("C++ concur: p99 tail flat under concurrent writer (<0.3ms)",
          max(r["p99"] for r in cc) < 0.3, f"max p99={max(r['p99'] for r in cc):.3f}ms")

fr = loadcpp("fresh.csv", ["system", "R", "tavg", "minr"])
if fr:
    svf = [r for r in fr if r["system"] == "StreamVec"][0]["tavg"]
    rb8 = [r for r in fr if r["R"] == 8000]
    check("C++ fresh: StreamVec time-avg recall ~0.977", approx(svf, 0.977, 0.01), f"{svf}")
    if rb8:
        check("C++ fresh: StreamVec > rebuild R=8000", svf > rb8[0]["tavg"],
              f"sv={svf} rb8={rb8[0]['tavg']}")

ch = loadcpp("churn.csv", ["system", "N", "tr", "recall", "qps"])
if ch:
    def chv(s, tr, f): return [r[f] for r in ch if r["system"] == s and r["tr"] == tr][0]
    check("C++ churn: repair beats no-repair at tau=0.4",
          chv("StreamVec", 0.4, "recall") > chv("StreamVec-NoRepair", 0.4, "recall"),
          f"sv={chv('StreamVec',0.4,'recall')} nr={chv('StreamVec-NoRepair',0.4,'recall')}")
    check("C++ churn: StreamVec QPS rises under churn (skip)",
          chv("StreamVec", 0.4, "qps") >= chv("StreamVec", 0.0, "qps"),
          f"{chv('StreamVec',0.0,'qps'):.0f}->{chv('StreamVec',0.4,'qps'):.0f}")

print("\n%d/%d claims verified" % (sum(c for _, c, _ in checks), len(checks)))
sys.exit(0 if all(c for _, c, _ in checks) else 1)
