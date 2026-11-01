"""Aggregate the v2 C++ runs (3 seeds) into summary.json, LaTeX tables and figures.
Inputs: results_cpp/v2/*.csv (written by cpp/bench modes e2e2, fresh2, churn, concur).
StreamVec rows come from the fix_* files (after the DirtySet/locking fixes)."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(H, "results_cpp", "v2"); FIG = os.path.join(H, "figures"); TAB = os.path.join(H, "tables")
plt.rcParams.update({"font.size": 7, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
os.makedirs(TAB, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(V2, f))
S = {}

# ---------- end-to-end ----------
NAMES = {"StreamVec": "\\sv", "StreamVec-C": "\\sv-C", "Skip-C": "\\textsc{Skip-C}", "Consolidate-0.15": "\\textsc{Consolidate}", "Append": "\\textsc{Append}",
         "ReplaceDeleted": "\\textsc{SlotReuse}", "Rebuild-R6000": "\\textsc{Rebuild}",
         "DeltaOverlay": "\\textsc{DeltaOverlay}", "StreamVec-NoRepair": "\\textsc{NoRepair}"}
ORDER = list(NAMES)
e2e = {}
for M in (16, 8):
    d = pd.read_csv(os.path.join(H, "results_cpp", "v3", f"M{M}_e2e_v2.csv"))   # single interleaved batch
    g = {}
    for sysn, x in d.groupby("system"):
        row = {}
        x40 = x[x.ef == 40]
        for col in ("tavg_recall_ef40", "update_tput", "memory_mb"):
            row[col] = [float(x40[col].mean()), float(x40[col].std(ddof=1))]
        for ef in (10, 40, 100):
            xe = x[x.ef == ef]
            row[f"recall_end_ef{ef}"] = [float(xe.recall_end.mean()), float(xe.recall_end.std(ddof=1))]
            row[f"qps_ef{ef}"] = [float(xe.query_qps.mean()), float(xe.query_qps.std(ddof=1))]
        row["n_seeds"] = int(x40.seed.nunique())
        g[sysn] = row
    e2e[f"M{M}"] = g
S["e2e"] = e2e

def pm(v, fmt): return (fmt % v[0]) + "{\\scriptsize$\\pm$" + (fmt % v[1]) + "}"
lines = ["\\begin{tabular}{l@{\\hspace{3pt}}r@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{\\hspace{4pt}}r}", "\\toprule",
         "System & \\shortstack{time-avg\\\\R@10} & \\shortstack{update\\\\kops/s} & \\shortstack{query\\\\kQPS} & \\shortstack{memory\\\\MB} \\\\"]
for M in (16, 8):
    lines += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\emph{{$M={M}$}}}}\\\\"]
    for s in ORDER:
        r = e2e[f"M{M}"][s]
        lines.append(f"{NAMES[s]} & {pm(r['tavg_recall_ef40'],'%.3f')} & {pm([r['update_tput'][0]/1e3, r['update_tput'][1]/1e3],'%.1f')}"
                     f" & {pm([r['qps_ef40'][0]/1e3, r['qps_ef40'][1]/1e3],'%.0f')} & {r['memory_mb'][0]:.1f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "cpp_e2e.tex"), "w").write("\n".join(lines) + "\n")

# ---------- scheduler ablation (matched budget, M=8) ----------
d = rd("fix_sched_e2e_v2.csv"); d0 = rd("sched_e2e_v2.csv"); d0 = d0[d0.system == "StreamVec-NoRepair"]
d = pd.concat([d, d0]); x = d[d.ef == 40]; sched = {}
for sysn, y in x.groupby("system"):
    sched[sysn] = {"tavg": [float(y.tavg_recall_ef40.mean()), float(y.tavg_recall_ef40.std(ddof=1))],
                   "tput": [float(y.update_tput.mean()), float(y.update_tput.std(ddof=1))],
                   "recall_end": [float(y.recall_end.mean()), float(y.recall_end.std(ddof=1))]}
S["sched"] = sched
SN = [("StreamVec", "Priority (deficit)"), ("StreamVec-FIFO", "FIFO"), ("StreamVec-Random", "Random"),
      ("StreamVec-Eager", "Eager (all dirty)"), ("StreamVec-NoRepair", "No repair")]
lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Policy & time-avg R@10 & final R@10 & update ops/s \\\\", "\\midrule"]
for k, nm in SN:
    r = sched[k]
    lines.append(f"{nm} & {pm(r['tavg'],'%.4f')} & {pm(r['recall_end'],'%.4f')} & {pm(r['tput'],'%.0f')} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "cpp_sched.tex"), "w").write("\n".join(lines) + "\n")

# ---------- freshness ----------
f = rd("fresh_v2.csv"); f = f[~((f.system == "StreamVec") & (f.R == 0) & False)]
fx = rd("fix_fresh_v2.csv"); fx["system"] = "StreamVec-NoTrigger"
# fresh_v2 'StreamVec' rows: first 3 are the no-trigger runs before the fix (dropped), later 3 the trigger runs
sv_rows = f[f.system == "StreamVec"]
f = f[f.system != "StreamVec"]
sv_trig = sv_rows.iloc[-3:]
f = pd.concat([f, sv_trig, fx])
fresh = {}
for sysn, y in f.groupby("system"):
    ser = np.array([[float(v) for v in s.strip(";").split(";")] for s in y.series])
    fresh[sysn] = {"tavg": [float(y.time_avg_recall.mean()), float(y.time_avg_recall.std(ddof=1))],
                   "min": [float(y.min_recall.mean()), float(y.min_recall.std(ddof=1))],
                   "tput": [float(y.update_tput.mean()), float(y.update_tput.std(ddof=1))],
                   "series_mean": ser.mean(0).tolist(), "n": int(len(y))}
S["fresh"] = fresh
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [2.2, 1]})
x = np.arange(1, len(fresh["StreamVec"]["series_mean"]) + 1) * 2
sty = [("StreamVec", "StreamVec", "#d62728", "-"), ("Consolidate-0.15", "Consolidate", "#1f77b4", "-"),
       ("Append", "Append", "#9467bd", "--"), ("DeltaOverlay", "DeltaOverlay", "#8c564b", ":"),
       ("Rebuild-R2000", "Rebuild R=2k", "#2ca02c", "-"), ("Rebuild-R8000", "Rebuild R=8k", "#17becf", "-")]
for k, lab, c, ls in sty:
    axes[0].plot(x, fresh[k]["series_mean"], ls, color=c, lw=1.0, label=lab)
axes[0].set_xlabel("operations (k)"); axes[0].set_ylabel("Recall@10 (live set)"); axes[0].set_ylim(0.75, 1.0)
axes[0].legend(ncol=3, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
bars = [("StreamVec", "SV"), ("StreamVec-NoTrigger", "SV-noT"), ("Consolidate-0.15", "Cons"), ("Append", "Apnd"),
        ("DeltaOverlay", "Delta"), ("Rebuild-R2000", "R2k"), ("Rebuild-R4000", "R4k"), ("Rebuild-R8000", "R8k")]
vals = [fresh[k]["tavg"][0] for k, _ in bars]; errs = [fresh[k]["tavg"][1] for k, _ in bars]
cols = ["#d62728", "#ff9896", "#1f77b4", "#9467bd", "#8c564b", "#2ca02c", "#98df8a", "#17becf"]
for i, (v_, e_, c_) in enumerate(zip(vals, errs, cols)):
    axes[1].errorbar([i], [v_], yerr=[e_], fmt="o", ms=4, color=c_, capsize=2)
axes[1].set_xticks(range(len(bars))); axes[1].set_xticklabels([b for _, b in bars])
axes[1].set_ylim(0.85, 1.0); axes[1].set_ylabel("time-avg Recall@10"); axes[1].tick_params(axis="x", rotation=60, labelsize=5.5)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "cpp_fresh_v2.pdf")); fig.savefig(os.path.join(FIG, "cpp_fresh_v2.png"), dpi=150); plt.close(fig)

# ---------- churn (delete-only) ----------
c = rd("churn.csv"); churn = {}
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.45))
cs = {"StreamVec": ("#d62728", "StreamVec"), "StreamVec-NoRepair": ("#7f7f7f", "NoRepair"),
      "Append": ("#9467bd", "Append"), "Consolidate-0.15": ("#1f77b4", "Consolidate")}
for k, g in c.groupby("system"):
    g = g.sort_values("tombstone_ratio"); base = float(g[g.tombstone_ratio == 0].query_qps.iloc[0])
    churn[k] = {"frac": g.tombstone_ratio.tolist(), "recall": g.recall_at_10.tolist(), "qps": g.query_qps.tolist(),
                "qps_norm": (g.query_qps / base).tolist()}
    col, lab = cs[k]
    axes[0].plot(g.tombstone_ratio, g.recall_at_10, "o-", ms=2, lw=0.9, color=col, label=lab)
    axes[1].plot(g.tombstone_ratio, g.query_qps / base, "o-", ms=2, lw=0.9, color=col, label=lab)
S["churn"] = churn
axes[0].set_xlabel("deleted fraction", fontsize=6); axes[0].set_ylabel("Recall@10", fontsize=6)
axes[1].set_xlabel("deleted fraction", fontsize=6); axes[1].set_ylabel("QPS / QPS(0)", fontsize=6)
axes[0].legend(fontsize=4.5, loc="lower left")
for a in axes: a.tick_params(labelsize=5)
fig.tight_layout(pad=0.2, w_pad=0.6)
fig.savefig(os.path.join(FIG, "cpp_churn_v2.pdf")); fig.savefig(os.path.join(FIG, "cpp_churn_v2.png"), dpi=150); plt.close(fig)

# ---------- concurrency ----------
cc2 = rd("concur2.csv"); cc = cc2[cc2.writer == 1].sort_values("query_threads"); cc0 = cc2[cc2.writer == 0].sort_values("query_threads")
S["concur"] = cc.to_dict(orient="list")
fig, ax1 = plt.subplots(figsize=(2.3, 1.45))
ax1.plot(cc.query_threads, cc.query_qps / 1e3, "o-", ms=3, lw=1, color="#d62728", label="with writer")
ax1.plot(cc0.query_threads, cc0.query_qps / 1e3, "o:", ms=3, lw=1, color="#ff9896", label="no writer"); ax1.legend(fontsize=5, loc="upper left", title="solid: kQPS, dashed: p99", title_fontsize=5)
ax1.set_xlabel("query threads", fontsize=6); ax1.set_ylabel("query kQPS", color="#d62728", fontsize=6); ax1.set_xticks(cc.query_threads)
ax2 = ax1.twinx(); ax2.plot(cc.query_threads, cc.p99_ms, "s--", ms=3, lw=1, color="#1f77b4"); ax2.plot(cc0.query_threads, cc0.p99_ms, "s:", ms=3, lw=1, color="#aec7e8")
ax2.set_ylabel("p99 latency (ms)", color="#1f77b4", fontsize=6); ax2.set_ylim(0, 0.3)
ax1.tick_params(labelsize=5); ax2.tick_params(labelsize=5)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(FIG, "cpp_concur_v2.pdf")); fig.savefig(os.path.join(FIG, "cpp_concur_v2.png"), dpi=150); plt.close(fig)

sc = rd("scale.csv").sort_values("N"); S["scale"] = sc.to_dict(orient="list")
fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.2)); Nk = sc.N / 1e3
axes[0].plot(Nk, sc.insert_ms, "o-", ms=2.5, lw=1, color="#d62728", label="insert")
axes[0].plot(Nk, sc.query_p50_ms, "s-", ms=2.5, lw=1, color="#1f77b4", label="query p50")
axes[0].set_ylabel("ms", fontsize=6); axes[0].legend(fontsize=4.5, loc="upper left")
axes[1].plot(Nk, sc.memory_mb, "o-", ms=2.5, lw=1, color="#8c564b"); axes[1].set_ylabel("memory (MB)", fontsize=6)
axes[2].plot(Nk, sc.par_qps / 1e3, "^-", ms=2.5, lw=1, color="#2ca02c"); axes[2].set_ylabel("kQPS (4 thr.)", fontsize=6)
for a in axes: a.tick_params(labelsize=5); a.set_xlabel("N (k)", fontsize=6)
fig.tight_layout(pad=0.2, w_pad=0.4)
fig.savefig(os.path.join(FIG, "cpp_scale_v2.pdf")); fig.savefig(os.path.join(FIG, "cpp_scale_v2.png"), dpi=150); plt.close(fig)



# ---------- degree sweep, budget sweep, recall-QPS frontier ----------
def tav(df, name):
    x = df[(df.system == name) & (df.ef == 40)]
    return float(x.tavg_recall_ef40.mean()), float(x.tavg_recall_ef40.std(ddof=1)), float(x.update_tput.mean())
deg = {}
for M in (6, 8, 12, 16, 24):
    if M in (8, 16):
        d = pd.concat([rd(f"M{M}_e2e_v2.csv"), rd(f"fix_M{M}_e2e_v2.csv")]); d = d[~((d.system == "StreamVec") & (d.index < 0))]
        d = pd.concat([rd(f"M{M}_e2e_v2.csv").query("system != 'StreamVec'"), rd(f"fix_M{M}_e2e_v2.csv")])
    else:
        d = rd(f"deg_M{M}_e2e_v2.csv")
    deg[M] = {k: tav(d, k) for k in ("StreamVec", "StreamVec-NoRepair", "Consolidate-0.15")}
S["degree"] = {str(k): v for k, v in deg.items()}
bud = {}
b0 = rd("budget_e2e_v2.csv"); b8 = rd("fix_sched_e2e_v2.csv")
for b in (2, 4, 16, 32, 64):
    x = b0[(b0.ef == 40) & (b0.system == "StreamVec")]
    # budget file holds only StreamVec rows; split by row order (5 budgets x 3 seeds)
bud_rows = b0[b0.ef == 40].reset_index(drop=True)
order = [2, 4, 16, 32, 64]
for i, b in enumerate(order):
    x = bud_rows.iloc[[i + 5 * j for j in range(3)]]
    bud[b] = (float(x.tavg_recall_ef40.mean()), float(x.tavg_recall_ef40.std(ddof=1)), float(x.update_tput.mean()))
bud[0] = tuple(sched["StreamVec-NoRepair"]["tavg"]) + (sched["StreamVec-NoRepair"]["tput"][0],)
bud[8] = tuple(sched["StreamVec"]["tavg"]) + (sched["StreamVec"]["tput"][0],)
bud["eager"] = tuple(sched["StreamVec-Eager"]["tavg"]) + (sched["StreamVec-Eager"]["tput"][0],)
S["budget"] = {str(k): v for k, v in bud.items()}

fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
Ms = [6, 8, 12, 16, 24]
for k, lab, c, mk in [("StreamVec", "StreamVec", "#d62728", "o"), ("Consolidate-0.15", "Consolidate", "#1f77b4", "s"),
                      ("StreamVec-NoRepair", "NoRepair", "#7f7f7f", "^")]:
    axes[0].errorbar(Ms, [deg[m][k][0] for m in Ms], yerr=[deg[m][k][1] for m in Ms], fmt=mk + "-", ms=3, lw=1, color=c, label=lab, capsize=1.5)
    axes[1].plot(Ms, [deg[m][k][2] / 1e3 for m in Ms], mk + "-", ms=3, lw=1, color=c, label=lab)
axes[0].set_xlabel("degree $M$"); axes[0].set_ylabel("time-avg Recall@10"); axes[0].legend(fontsize=5.5, loc="lower right")
axes[1].set_xlabel("degree $M$"); axes[1].set_ylabel("update kops/s"); axes[1].legend(fontsize=5.5, loc="upper right")
for a in axes[:2]: a.set_xticks(Ms)
keys = [0, 2, 4, 8, 16, 32, 64, "eager"]; xs = list(range(len(keys)))
axes[2].errorbar(xs, [bud[b][0] for b in keys], yerr=[bud[b][1] for b in keys], fmt="o-", ms=3, lw=1, color="#d62728", capsize=1.5)
axes[2].set_xticks(xs); axes[2].set_xticklabels([str(b) if b != "eager" else "all" for b in keys])
axes[2].set_xlabel("repair budget $b$ per 128 ops ($M{=}8$)"); axes[2].set_ylabel("time-avg Recall@10", color="#d62728")
a2 = axes[2].twinx(); a2.plot(xs, [bud[b][2] / 1e3 for b in keys], "s--", ms=2.5, lw=0.8, color="#1f77b4")
a2.set_ylabel("update kops/s", color="#1f77b4"); a2.tick_params(labelsize=6); a2.grid(False)
for a in axes: a.tick_params(labelsize=6)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(os.path.join(FIG, "cpp_degree_budget.pdf")); fig.savefig(os.path.join(FIG, "cpp_degree_budget.png"), dpi=150); plt.close(fig)

# ---------- round 2: 2x2 decomposition, single-thread, clean matched budget, concurrency ----------
def agg(df, name, ef=40):
    x = df[(df.system == name) & (df.ef == ef)]
    return {"tavg": [float(x.tavg_recall_ef40.mean()), float(x.tavg_recall_ef40.std(ddof=1))],
            "tput": [float(x.update_tput.mean()), float(x.update_tput.std(ddof=1))],
            "recall_end": [float(x.recall_end.mean()), float(x.recall_end.std(ddof=1))],
            "qps": [float(x.query_qps.mean()), float(x.query_qps.std(ddof=1))], "n": int(len(x))}
dec = {}
for M in (6, 8, 12, 16, 24):
    d = rd(f"r2_skipc_M{M}_e2e_v2.csv")
    dec[str(M)] = {"StreamVec-C": agg(d, "StreamVec-C"), "Skip-C": agg(d, "Skip-C")}
S["decomp"] = dec
one = {}
for M in (16, 8):
    d = rd(f"r2_1thr_M{M}_e2e_v2.csv")
    one[str(M)] = {k: agg(d, k) for k in ("StreamVec", "Consolidate-0.15", "StreamVec-C", "Skip-C")}
S["single_thread"] = one
d = rd("r2_sched_e2e_v2.csv"); x = d[d.ef == 40].reset_index(drop=True)
lab = ["Priority", "FIFO", "Random", "Eager", "None"]; cs = {}
for i, l in enumerate(lab):
    y = x.iloc[[i + 5 * j for j in range(3)]]
    cs[l] = {"tavg": [float(y.tavg_recall_ef40.mean()), float(y.tavg_recall_ef40.std(ddof=1))],
             "recall_end": [float(y.recall_end.mean()), float(y.recall_end.std(ddof=1))],
             "tput": [float(y.update_tput.mean()), float(y.update_tput.std(ddof=1))], "systems": list(y.system)}
S["sched_clean"] = cs
lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Policy & time-avg R@10 & final R@10 & updates/s \\\\", "\\midrule"]
for l, nm in zip(lab, ["Priority (deficit)", "FIFO", "Random", "Eager (all dirty)", "No repair ($b{=}0$)"]):
    r = cs[l]; lines.append(f"{nm} & {pm(r['tavg'],'%.4f')} & {pm(r['recall_end'],'%.4f')} & {pm(r['tput'],'%.0f')} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "cpp_sched.tex"), "w").write("\n".join(lines) + "\n")
c2 = rd("concur2.csv"); S["concur2"] = c2.to_dict(orient="list")

json.dump(S, open(os.path.join(V2, "summary.json"), "w"), indent=1)
print(json.dumps({k: {s: (v.get('tavg_recall_ef40'), v.get('update_tput'), v.get('qps_ef40'), v.get('memory_mb')) for s, v in e2e[k].items()} for k in e2e}, indent=0))
print({k: (v['tavg'], v['tput']) for k, v in sched.items()})
print({k: (v['tavg'], v['min'], v['n']) for k, v in fresh.items()})
print({k: (v['recall'], [round(q,3) for q in v['qps_norm']]) for k, v in churn.items()})
