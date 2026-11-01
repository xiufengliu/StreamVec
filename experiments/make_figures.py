"""Generate figures (PDF + PNG) and LaTeX tables from results CSVs.

Reads only real result files under results/; never fabricates. Any missing
experiment is skipped with a warning so partial runs still produce what exists.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
MARK = {"StreamVec": ("o", "#d62728"), "HNSW-Append": ("s", "#1f77b4"),
        "HNSW-Rebuild": ("^", "#2ca02c"), "FAISS-IVF-Rebuild": ("D", "#9467bd"),
        "Python-Rebuild": ("v", "#ff7f0e"), "StreamVec-NoRepair": ("x", "#7f7f7f")}


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print("wrote", name)


def _exists(p):
    if not os.path.exists(p):
        print("SKIP (missing):", p)
        return False
    return True


def fig_exp1():
    p = os.path.join(RES, "exp1", "pareto.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    for sysname, g in df.groupby("system"):
        g = g.sort_values("recall_at_10")
        m, c = MARK.get(sysname, ("o", None))
        lab = sysname.replace("FAISS-IVF-Rebuild", "FAISS-IVF").replace("Python-Rebuild", "Py-Rebuild")
        ax.plot(g["recall_at_10"], g["query_qps"] / 1000.0, marker=m, color=c, label=lab, lw=1, ms=3)
    ax.set_xlabel("Recall@10", fontsize=7); ax.set_ylabel("Query kQPS", fontsize=7)
    ax.legend(fontsize=5, handlelength=1.2, borderpad=0.3, labelspacing=0.25)
    ax.tick_params(labelsize=6, pad=1.5)
    fig.tight_layout(pad=0.2)
    _save(fig, "exp1_pareto")
    # table: update throughput per system
    upd = df.groupby("system")["update_throughput_ops_s"].first().reset_index()
    with open(os.path.join(TAB, "exp1_update_throughput.tex"), "w") as f:
        f.write("% auto-generated from results/exp1/pareto.csv\n")
        f.write("\\begin{tabular}{lr}\n\\toprule\nSystem & Update throughput (ops/s) \\\\\n\\midrule\n")
        for _, r in upd.iterrows():
            f.write(f"{r['system']} & {r['update_throughput_ops_s']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def fig_exp2():
    p = os.path.join(RES, "exp2", "recall_vs_tombstone.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7))
    for sysname, g in df.groupby("system"):
        m, c = MARK.get(sysname, ("o", None))
        lab = sysname.replace("HNSW-Append", "Append").replace("StreamVec-NoRepair", "NoRepair")
        axes[0].plot(g["tombstone_ratio"], g["recall_at_10"], marker=m, color=c, label=lab, lw=1, ms=2.5)
        # normalized QPS (each system relative to its own tomb=0 throughput) to
        # control for the Python-vs-C++ absolute-speed gap
        base_q = g[g["tombstone_ratio"] == 0.0]["query_qps"]
        norm = float(base_q.iloc[0]) if len(base_q) else float(g["query_qps"].iloc[0])
        axes[1].plot(g["tombstone_ratio"], g["query_qps"] / norm, marker=m, color=c, label=lab, lw=1, ms=2.5)
    axes[0].set_ylabel("Recall@10", fontsize=6); axes[0].set_title("Recall vs. churn", fontsize=6)
    axes[0].legend(fontsize=4.0, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    axes[1].set_ylabel("QPS (norm.)", fontsize=6); axes[1].set_title("Query cost vs. churn", fontsize=6)
    for ax in axes:
        ax.set_xlabel("Tombstone ratio", fontsize=6); ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    _save(fig, "exp2_recall_vs_tombstone")


def fig_exp3():
    ph = os.path.join(RES, "exp3", "ablation_history.csv")
    pf = os.path.join(RES, "exp3", "ablation_final.csv")
    if not _exists(ph):
        return
    df = pd.read_csv(ph)
    fig, ax = plt.subplots(figsize=(1.62, 1.5))
    colors = {"priority": "#d62728", "fifo": "#1f77b4", "random": "#2ca02c",
              "eager": "#ff7f0e", "norepair": "#7f7f7f"}
    for pol, g in df.groupby("policy"):
        g = g.sort_values("ops")
        ax.plot(g["ops"] / 1000.0, g["recall"], marker="o", ms=2, lw=0.9, label=pol, color=colors.get(pol))
    ax.set_xlabel("Stream ops (k)", fontsize=6); ax.set_ylabel("Recall@10", fontsize=6)
    ax.legend(fontsize=4.0, handlelength=1.0, borderpad=0.2, labelspacing=0.15)
    ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2)
    _save(fig, "exp3_repair_ablation")
    if _exists(pf):
        d = pd.read_csv(pf)
        with open(os.path.join(TAB, "exp3_ablation.tex"), "w") as f:
            f.write("% auto-generated from results/exp3/ablation_final.csv\n")
            f.write("\\begin{tabular}{lrrr}\n\\toprule\nPolicy & Final Recall@10 & Update tput (ops/s) & Repair dist. calls \\\\\n\\midrule\n")
            for _, r in d.iterrows():
                f.write(f"{r['policy']} & {r['final_recall']:.3f} & {r['update_throughput_ops_s']:.0f} & {int(r['repair_dist_calls'])} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")


def fig_exp4():
    p = os.path.join(RES, "exp4", "scalability.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    dN = df[df["axis"] == "N"].sort_values("N")
    # single-column, side-by-side, short: 1x3 panels at column width
    fig, axes = plt.subplots(1, 3, figsize=(3.3, 1.25))
    Nk = dN["N"] / 1000.0
    if len(dN):
        ax = axes[0]
        ax.plot(Nk, dN["insert_ms"], "o-", ms=2.5, lw=1, label="insert")
        ax.plot(Nk, dN["query_p50_ms"], "s-", ms=2.5, lw=1, label="query")
        ax.plot(Nk, dN["repair_ms_per_call"], "^-", ms=2.5, lw=1, label="repair")
        ax.set_ylabel("ms", fontsize=6); ax.set_title("Per-op latency", fontsize=6)
        ax.legend(fontsize=4.2, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
        ax = axes[1]
        ax.plot(Nk, dN["build_s"], "D-", ms=2.5, lw=1, color="#2ca02c")
        ax.set_ylabel("s", fontsize=6); ax.set_title("Rebuild cost", fontsize=6)
        ax = axes[2]
        ax.plot(Nk, dN["memory_mb"], "o-", ms=2.5, lw=1, color="#8c564b")
        ax.set_ylabel("MB", fontsize=6); ax.set_title("Memory", fontsize=6)
    for ax in axes:
        ax.tick_params(labelsize=5, pad=1.5); ax.set_xlabel("N (k)", fontsize=6)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    _save(fig, "exp4_scalability")
    # memory subplot
    if len(dN):
        fig2, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(dN["N"], dN["memory_mb"], "o-", color="#8c564b")
        ax.set_xlabel("Index size N"); ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory vs. N"); _save(fig2, "exp4_memory")


def fig_exp5():
    p = os.path.join(RES, "exp5", "repair_budget.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    df["bnum"] = df["repair_budget"].apply(lambda x: 10**9 if x == "all" else int(x))
    df = df.sort_values("bnum")
    fig, ax1 = plt.subplots(figsize=(1.62, 1.5))
    x = range(len(df))
    ax1.plot(x, df["final_recall"], "o-", ms=2, lw=0.9, color="#d62728")
    ax1.set_ylabel("Recall@10", color="#d62728", fontsize=6)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(df["repair_budget"].astype(str), fontsize=5)
    ax1.set_xlabel("Repair budget", fontsize=6)
    ax2 = ax1.twinx()
    ax2.plot(x, df["repair_dist_calls"], "s--", ms=2, lw=0.9, color="#1f77b4")
    ax2.set_ylabel("Repair dist. calls", color="#1f77b4", fontsize=6)
    ax1.tick_params(labelsize=5, pad=1.5); ax2.tick_params(labelsize=5, pad=1.5)
    ax2.yaxis.get_offset_text().set_fontsize(5)   # shrink the "1e7" multiplier label
    fig.tight_layout(pad=0.2)
    _save(fig, "exp5_repair_budget")


def fig_exp6():
    p = os.path.join(RES, "exp6", "rag_quality.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.6))
    for sysname, g in df.groupby("system"):
        m, c = MARK.get(sysname, ("o", None))
        lab = sysname.replace("HNSW-Rebuild", "Rebuild")
        axes[0].plot(g["step"] / 1000.0, g["recall_at_10"], marker=m, ms=2.5, lw=1, color=c, label=lab)
        axes[1].plot(g["step"] / 1000.0, g["ndcg_at_10"], marker=m, ms=2.5, lw=1, color=c, label=lab)
    axes[0].set_ylabel("Recall@10", fontsize=6); axes[0].set_title("Recall@10", fontsize=6)
    axes[1].set_ylabel("NDCG@10", fontsize=6); axes[1].set_title("NDCG@10", fontsize=6)
    axes[0].legend(fontsize=4.2, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    for ax in axes:
        ax.set_xlabel("Stream step (k)", fontsize=6); ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    _save(fig, "exp6_rag")


def fig_exp7():
    p = os.path.join(RES, "exp7", "degree_sensitivity.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    piv = df.pivot_table(index="M", columns="variant", values="final_recall")
    piv = piv.sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7))
    axes[0].plot(piv.index, piv["repair"], "o-", ms=2.5, lw=1, color="#d62728", label="repair")
    axes[0].plot(piv.index, piv["norepair"], "x--", ms=2.5, lw=1, color="#7f7f7f", label="no repair")
    axes[0].set_ylabel("Final Recall@10", fontsize=6); axes[0].set_title("Recall vs. degree", fontsize=6)
    axes[0].legend(fontsize=4.2, handlelength=1.2, borderpad=0.2, labelspacing=0.2)
    gain = (piv["repair"] - piv["norepair"]) * 100
    axes[1].bar(piv.index.astype(str), gain, color="#1f77b4")
    axes[1].set_ylabel("Repair gain (pts)", fontsize=6); axes[1].set_title("Gain shrinks with $M$", fontsize=6)
    for ax in axes:
        ax.set_xlabel("Graph degree $M$", fontsize=6); ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    _save(fig, "exp7_degree_sensitivity")


def fig_exp8():
    ps = os.path.join(RES, "exp8", "freshness_summary.csv")
    pt = os.path.join(RES, "exp8", "freshness_series.csv")
    if not _exists(ps):
        return
    s = pd.read_csv(ps)
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7))
    sv = s[s["system"] == "StreamVec"]["time_avg_recall"].iloc[0]
    rb = s[s["R"] > 0].sort_values("R")
    axes[0].axhline(sv, color="#d62728", lw=1.2, label="StreamVec")
    for name, col in [("StreamVec-NoRepair", "#7f7f7f"), ("HNSW-Append", "#1f77b4")]:
        row = s[s["system"] == name]
        if len(row):
            axes[0].axhline(float(row["time_avg_recall"].iloc[0]), color=col, lw=0.8, ls=":")
    axes[0].plot(rb["R"] / 1000.0, rb["time_avg_recall"], "s-", ms=2.5, lw=1, color="#2ca02c", label="Rebuild avg")
    axes[0].plot(rb["R"] / 1000.0, rb["min_recall"], "v--", ms=2.5, lw=1, color="#8c564b", label="Rebuild min")
    axes[0].set_xlabel("Rebuild interval (k ops)", fontsize=6); axes[0].set_ylabel("Time-avg Recall@10", fontsize=6)
    axes[0].set_title("Freshness vs. rebuild", fontsize=6)
    axes[0].legend(fontsize=4.0, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    if os.path.exists(pt):
        t = pd.read_csv(pt)
        for uid, g in t.groupby("unit_id"):
            if uid not in ("streamvec",) and not uid.startswith("rebuild_"):
                continue
            g = g.sort_values("ops")
            lab = "StreamVec" if uid == "streamvec" else uid.replace("rebuild_R", "R")
            axes[1].plot(g["ops"] / 1000.0, g["recall_at_10"], label=lab, lw=0.8)
        axes[1].set_xlabel("Stream ops (k)", fontsize=6); axes[1].set_ylabel("Recall@10", fontsize=6)
        axes[1].set_title("Rebuild sawtooth", fontsize=6)
        axes[1].legend(fontsize=3.8, ncol=2, handlelength=1.0, borderpad=0.2, labelspacing=0.2, columnspacing=0.6)
    for ax in axes:
        ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    _save(fig, "exp8_freshness")


def fig_exp9():
    p = os.path.join(RES, "exp9", "workload.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    fig, ax1 = plt.subplots(figsize=(2.6, 1.7))
    x = range(len(df))
    short = [m.replace("-heavy", "") for m in df["mix"]]
    ax1.bar([i - 0.2 for i in x], df["final_recall"], width=0.4, color="#d62728")
    ax1.set_ylabel("Final Recall@10", color="#d62728", fontsize=7); ax1.set_ylim(0.85, 1.0)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(short, rotation=20, fontsize=5.5)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], df["tombstone_ratio"], width=0.4, color="#1f77b4")
    ax2.set_ylabel("Tombstone ratio", color="#1f77b4", fontsize=7)
    ax1.tick_params(axis="y", labelsize=6, pad=1.5); ax2.tick_params(labelsize=6, pad=1.5)
    fig.tight_layout(pad=0.2)
    _save(fig, "exp9_workload")


def fig_exp10():
    p = os.path.join(RES, "exp10", "trigger_series.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7))
    col = {"trigger_on": "#d62728", "trigger_off": "#7f7f7f"}
    lab = {"trigger_on": "ON", "trigger_off": "OFF"}
    for m, g in df.groupby("mode"):
        g = g.sort_values("ops")
        axes[0].plot(g["ops"] / 1000.0, g["tombstone_ratio"], "o-", ms=2.5, lw=1, color=col.get(m), label=lab.get(m, m))
        axes[1].plot(g["ops"] / 1000.0, g["memory_mb"], "o-", ms=2.5, lw=1, color=col.get(m), label=lab.get(m, m))
    axes[0].set_ylabel("Tombstone ratio", fontsize=6); axes[0].set_title("Trigger bounds $\\tau$", fontsize=6)
    axes[0].legend(fontsize=4.5, title="trigger", title_fontsize=4.5, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    axes[1].set_ylabel("Memory (MB)", fontsize=6); axes[1].set_title("...and memory", fontsize=6)
    for ax in axes:
        ax.set_xlabel("Stream ops (k)", fontsize=6); ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    _save(fig, "exp10_trigger")


def fig_exp11():
    p = os.path.join(RES, "exp11", "glove_recall_vs_tombstone.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7))
    for sysname, g in df.groupby("system"):
        m, c = MARK.get(sysname, ("o", None))
        lab = sysname.replace("HNSW-Append", "Append").replace("StreamVec-NoRepair", "NoRepair")
        axes[0].plot(g["tombstone_ratio"], g["recall_at_10"], marker=m, color=c, label=lab, lw=1, ms=2.5)
        b = g[g["tombstone_ratio"] == 0.0]["query_qps"]
        norm = float(b.iloc[0]) if len(b) else float(g["query_qps"].iloc[0])
        axes[1].plot(g["tombstone_ratio"], g["query_qps"] / norm, marker=m, color=c, label=lab, lw=1, ms=2.5)
    axes[0].set_ylabel("Recall@10", fontsize=6); axes[0].set_title("GloVe: recall", fontsize=6)
    axes[0].legend(fontsize=4.0, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    axes[1].set_ylabel("QPS (norm.)", fontsize=6); axes[1].set_title("GloVe: query cost", fontsize=6)
    for ax in axes:
        ax.set_xlabel("Tombstone ratio", fontsize=6); ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    _save(fig, "exp11_glove")


def fig_q9_combined():
    # Q9: combine RAG (exp6) and GloVe (exp11) into one 1x4 row, single column.
    p6 = os.path.join(RES, "exp6", "rag_quality.csv")
    p11 = os.path.join(RES, "exp11", "glove_recall_vs_tombstone.csv")
    if not (_exists(p6) and _exists(p11)):
        return
    rag = pd.read_csv(p6); glo = pd.read_csv(p11)
    fig, axes = plt.subplots(1, 4, figsize=(3.3, 1.05))
    # (a),(b) RAG recall / ndcg over stream step
    for sysname, g in rag.groupby("system"):
        m, c = MARK.get(sysname, ("o", None))
        lab = sysname.replace("HNSW-Rebuild", "Rebuild")
        axes[0].plot(g["step"] / 1000.0, g["recall_at_10"], marker=m, ms=2, lw=0.8, color=c, label=lab)
        axes[1].plot(g["step"] / 1000.0, g["ndcg_at_10"], marker=m, ms=2, lw=0.8, color=c)
    axes[0].set_title("RAG R@10", fontsize=5.5); axes[1].set_title("RAG NDCG", fontsize=5.5)
    axes[0].legend(fontsize=3.4, handlelength=0.9, borderpad=0.15, labelspacing=0.12)
    for a in (axes[0], axes[1]):
        a.set_xlabel("step (k)", fontsize=5.5)
    # (c),(d) GloVe recall / normalized QPS vs tombstone
    for sysname, g in glo.groupby("system"):
        m, c = MARK.get(sysname, ("o", None))
        lab = sysname.replace("HNSW-Append", "Apnd").replace("StreamVec-NoRepair", "NoRep").replace("StreamVec", "SV")
        axes[2].plot(g["tombstone_ratio"], g["recall_at_10"], marker=m, ms=2, lw=0.8, color=c, label=lab)
        b = g[g["tombstone_ratio"] == 0.0]["query_qps"]
        norm = float(b.iloc[0]) if len(b) else float(g["query_qps"].iloc[0])
        axes[3].plot(g["tombstone_ratio"], g["query_qps"] / norm, marker=m, ms=2, lw=0.8, color=c)
    axes[2].set_title("GloVe R@10", fontsize=5.5); axes[3].set_title("GloVe QPS", fontsize=5.5)
    axes[2].legend(fontsize=3.2, handlelength=0.9, borderpad=0.15, labelspacing=0.12)
    for a in (axes[2], axes[3]):
        a.set_xlabel("tomb. ratio", fontsize=5.5)
    for ax in axes:
        ax.tick_params(labelsize=4.2, pad=1)
    fig.tight_layout(pad=0.15, w_pad=0.5)
    _save(fig, "q9_combined")


def fig_exp12():
    p = os.path.join(RES, "exp12", "param_sensitivity.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p)
    params = ["tau_th", "delta_rep", "P", "ef_r"]
    titles = {"tau_th": r"$\tau_{th}$", "delta_rep": r"$\Delta_{rep}$",
              "P": "P", "ef_r": r"$ef_r$"}
    # 1x4 row: red recall (shared left axis, numbers only on the leftmost panel);
    # blue dashed = normalized cost trend (no right ticks, keeps panels wide).
    fig, axes = plt.subplots(1, 4, figsize=(3.3, 1.15), sharey=True)
    for i, (ax, par) in enumerate(zip(axes, params)):
        g = df[df["param"] == par].sort_values("value")
        ax.plot(g["value"], g["final_recall"], "o-", ms=2.5, lw=0.9, color="#d62728",
                label="recall")
        ax.set_ylim(0.95, 0.985)
        ax.set_xlabel(titles[par], fontsize=7)
        # normalize cost into the recall band so the trend is visible without a 2nd axis
        cost = (g["rebuilds"] if par == "tau_th" else g["update_throughput_ops_s"]).astype(float)
        cn = (cost - cost.min()) / (cost.max() - cost.min() + 1e-9)
        ax.plot(g["value"], 0.951 + cn * (0.984 - 0.951), "s--", ms=2.5, lw=0.9,
                color="#1f77b4", label="cost")
        xv = list(g["value"])
        ax.set_xticks([xv[0], xv[-1]])
        ax.tick_params(labelsize=5, pad=1.5)
    axes[0].set_ylabel("Recall@10", fontsize=6.5)
    axes[0].set_yticks([0.95, 0.96, 0.97, 0.98])
    axes[0].legend(fontsize=4.2, handlelength=1.1, borderpad=0.2, labelspacing=0.15, loc="lower right")
    fig.tight_layout(pad=0.15, w_pad=0.6)
    _save(fig, "exp12_param_sensitivity")


def fig_exp13():
    pa = os.path.join(RES, "exp13", "adversarial.csv")
    ph = os.path.join(RES, "exp13", "hotness.csv")
    if not _exists(pa):
        return
    da = pd.read_csv(pa)
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.7))
    style = {("hub", "repair"): ("o-", "#d62728"), ("hub", "norepair"): ("o--", "#ff7f0e"),
             ("random", "repair"): ("s-", "#2ca02c"), ("random", "norepair"): ("s--", "#7f7f7f")}
    for (tg, var), g in da.groupby(["target", "variant"]):
        g = g.sort_values("tombstone_ratio")
        st, c = style.get((tg, var), ("o-", None))
        axes[0].plot(g["tombstone_ratio"], g["recall_at_10"], st, color=c, ms=2.5, lw=1,
                     label=f"{tg}/{var}")
    axes[0].set_xlabel("Tombstone ratio", fontsize=6); axes[0].set_ylabel("Recall@10", fontsize=6)
    axes[0].set_title("Hub vs. random deletion", fontsize=6)
    axes[0].legend(fontsize=4.0, handlelength=1.2, borderpad=0.2, labelspacing=0.2)
    if os.path.exists(ph):
        dh = pd.read_csv(ph)
        x = range(len(dh))
        axes[1].bar([i - 0.2 for i in x], dh["hot_recall"], width=0.4, color="#d62728", label="hot")
        axes[1].bar([i + 0.2 for i in x], dh["uniform_recall"], width=0.4, color="#1f77b4", label="unif")
        axes[1].set_xticks(list(x)); axes[1].set_xticklabels(dh["policy"], fontsize=5)
        axes[1].set_ylim(0.92, 0.965); axes[1].set_ylabel("Recall@10", fontsize=6)
        axes[1].set_title("Hotness scheduling", fontsize=6)
        axes[1].legend(fontsize=4.5, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    for ax in axes:
        ax.tick_params(labelsize=5, pad=1.5)
    fig.tight_layout(pad=0.2, w_pad=1.2)
    _save(fig, "exp13_adversarial_hotness")


def fig_exp14():
    p = os.path.join(RES, "exp14", "microbench.csv")
    if not _exists(p):
        return
    df = pd.read_csv(p).sort_values("tombstone_ratio")
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    ax.plot(df["tombstone_ratio"], df["nchecks_per_query"], "s-", ms=3, lw=1, color="#1f77b4", label="adjacency scans")
    ax.plot(df["tombstone_ratio"], df["dist_per_query"], "o-", ms=3, lw=1, color="#d62728", label="distance evals")
    ax.plot(df["tombstone_ratio"], df["heapops_per_query"], "^-", ms=3, lw=1, color="#2ca02c", label="heap ops")
    ax.set_xlabel("Tombstone ratio", fontsize=7); ax.set_ylabel("Ops per query", fontsize=7)
    ax.legend(fontsize=5, handlelength=1.2, borderpad=0.3, labelspacing=0.25)
    ax.tick_params(labelsize=6, pad=1.5)
    fig.tight_layout(pad=0.2)
    _save(fig, "exp14_microbench")


if __name__ == "__main__":
    for fn in (fig_exp1, fig_exp2, fig_exp3, fig_exp4, fig_exp5, fig_exp6,
               fig_exp7, fig_exp8, fig_exp9, fig_exp10, fig_exp11,
               fig_q9_combined, fig_exp12, fig_exp13, fig_exp14):
        try:
            fn()
        except Exception as e:
            print("ERROR in", fn.__name__, ":", e)
    print("figures done")
