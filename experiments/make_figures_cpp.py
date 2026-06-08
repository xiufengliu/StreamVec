"""Figures from the C++ harness results (results_cpp/*.csv). Vector PDF output."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results_cpp")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})


COLS = {
    "e2e.csv": ["system", "N", "ef", "recall_at_10", "query_qps", "update_tput", "memory_mb"],
    "scale.csv": ["N", "build_ins_per_s", "insert_ms", "query_p50_ms", "par_qps", "memory_mb"],
    "concur.csv": ["query_threads", "N", "query_qps", "p50_ms", "p99_ms"],
    "fresh.csv": ["system", "R", "time_avg_recall", "min_recall"],
    "churn.csv": ["system", "N", "tombstone_ratio", "recall_at_10", "query_qps"],
}


def load(name):
    p = os.path.join(RES, name)
    # files were written without a header row; assign known names
    df = pd.read_csv(p, header=None, names=COLS[name])
    # drop any accidental header-as-data row
    df = df[df[COLS[name][0]].astype(str) != COLS[name][0]]
    for c in COLS[name][1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save(fig, n):
    fig.savefig(os.path.join(FIG, n + ".pdf")); fig.savefig(os.path.join(FIG, n + ".png"), dpi=120)
    plt.close(fig); print("wrote", n)


def fig_e2e():
    df = load("e2e.csv")
    d = df[df.ef == 40].copy()
    fig, axes = plt.subplots(1, 3, figsize=(3.3, 1.45))
    order = ["StreamVec", "DeltaOverlay", "Rebuild-R6000", "Append", "ReplaceDeleted"]
    d = d.set_index("system").reindex([s for s in order if s in set(d.system)]).reset_index()
    col = {"StreamVec": "#d62728", "DeltaOverlay": "#9467bd", "Rebuild-R6000": "#2ca02c",
           "Append": "#1f77b4", "ReplaceDeleted": "#ff7f0e"}
    cs = [col.get(s, "#333") for s in d.system]
    short = ["SV", "Delta", "Rbld", "Apnd", "Slot"][:len(d.system)]
    axes[0].bar(short, d.recall_at_10, color=cs); axes[0].set_ylim(0.95, 0.99)
    axes[0].set_title("Recall@10", fontsize=6)
    axes[1].bar(short, d.update_tput / 1000.0, color=cs)
    axes[1].set_title("Update kops/s", fontsize=6)
    axes[2].bar(short, d.memory_mb, color=cs); axes[2].set_title("Memory (MB)", fontsize=6)
    for ax in axes:
        ax.tick_params(axis="x", rotation=40, labelsize=4.6, pad=1)
        ax.tick_params(axis="y", labelsize=5, pad=1)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    save(fig, "cpp_e2e")


def fig_scale():
    df = load("scale.csv").sort_values("N")
    fig, axes = plt.subplots(1, 3, figsize=(3.3, 1.25))
    Nk = df.N / 1000.0
    axes[0].plot(Nk, df.insert_ms, "o-", ms=2.5, lw=1, color="#d62728", label="insert")
    axes[0].plot(Nk, df.query_p50_ms, "s-", ms=2.5, lw=1, color="#1f77b4", label="query")
    axes[0].set_ylabel("ms", fontsize=6); axes[0].set_title("Per-op latency", fontsize=6)
    axes[0].legend(fontsize=4.2, handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    axes[1].plot(Nk, df.memory_mb, "o-", ms=2.5, lw=1, color="#8c564b")
    axes[1].set_ylabel("MB", fontsize=6); axes[1].set_title("Memory", fontsize=6)
    axes[2].plot(Nk, df.par_qps / 1000.0, "^-", ms=2.5, lw=1, color="#2ca02c")
    axes[2].set_ylabel("kQPS", fontsize=6); axes[2].set_title("Query tput (4 thr)", fontsize=6)
    for ax in axes:
        ax.tick_params(labelsize=5, pad=1.5); ax.set_xlabel("N (k)", fontsize=6)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    save(fig, "cpp_scale")


def fig_concur():
    df = load("concur.csv").sort_values("query_threads")
    fig, ax1 = plt.subplots(figsize=(2.6, 1.7))
    ax1.plot(df.query_threads, df.query_qps / 1000.0, "o-", ms=3, lw=1, color="#d62728")
    ax1.set_xlabel("Query threads", fontsize=7); ax1.set_ylabel("Query kQPS", color="#d62728", fontsize=7)
    ax1.set_xticks(df.query_threads)
    ax2 = ax1.twinx()
    ax2.plot(df.query_threads, df.p99_ms, "s--", ms=3, lw=1, color="#1f77b4")
    ax2.set_ylabel("p99 latency (ms)", color="#1f77b4", fontsize=7); ax2.set_ylim(0, 0.4)
    ax1.tick_params(labelsize=6, pad=1.5); ax2.tick_params(labelsize=6, pad=1.5)
    fig.tight_layout(pad=0.2)
    save(fig, "cpp_concur")


def fig_fresh():
    df = load("fresh.csv")
    sv = df[df.system == "StreamVec"].time_avg_recall.iloc[0]
    rb = df[df.R > 0].sort_values("R")
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.axhline(sv, color="#d62728", lw=2, label="StreamVec (fresh)")
    ax.plot(rb.R, rb.time_avg_recall, "s-", color="#2ca02c", label="Periodic rebuild")
    ax.plot(rb.R, rb.min_recall, "v--", color="#8c564b", label="Rebuild min")
    ax.set_xlabel("Rebuild interval R (ops)"); ax.set_ylabel("Time-averaged Recall@10")
    ax.set_title("Freshness (C++, SIFT)"); ax.legend(fontsize=8)
    save(fig, "cpp_fresh")


def fig_churn():
    df = load("churn.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    col = {"StreamVec": "#d62728", "StreamVec-NoRepair": "#7f7f7f", "Append": "#1f77b4"}
    for s, g in df.groupby("system"):
        g = g.sort_values("tombstone_ratio")
        axes[0].plot(g.tombstone_ratio, g.recall_at_10, "o-", color=col.get(s), label=s)
        n = g[g.tombstone_ratio == 0].query_qps.iloc[0]
        axes[1].plot(g.tombstone_ratio, g.query_qps / n, "o-", color=col.get(s), label=s)
    axes[0].set_xlabel("Tombstone ratio"); axes[0].set_ylabel("Recall@10")
    axes[0].set_title("Recall vs. churn"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Tombstone ratio"); axes[1].set_ylabel("Query QPS (norm.)")
    axes[1].set_title("Query cost vs. churn"); axes[1].legend(fontsize=8)
    save(fig, "cpp_churn")


if __name__ == "__main__":
    for fn in (fig_e2e, fig_scale, fig_concur, fig_fresh, fig_churn):
        try: fn()
        except Exception as e: print("ERR", fn.__name__, e)
    print("done")
