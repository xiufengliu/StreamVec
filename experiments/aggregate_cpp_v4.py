"""Aggregate v4 C++ runs (bench mode e2e3): GloVe and delete-heavy generality, time-averaged
single-thread query throughput, repair diagnostics. Writes results_cpp/v4/summary_v4.json and
tables/cpp_general.tex."""
import json, os
import pandas as pd
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(H, "results_cpp", "v4"); TAB = os.path.join(H, "tables")
os.makedirs(TAB, exist_ok=True)
def rd(f): return pd.read_csv(os.path.join(V4, f))
S = {}
sets = [("glove", "glove_e2e3.csv"), ("del16", "delheavy_M16_e2e3.csv"), ("del8", "delheavy_M8_e2e3.csv"),
        ("mix16", "mixed_M16_e2e3.csv"), ("mix8", "mixed_M8_e2e3.csv")]
for k, f in sets:
    d = rd(f); g = {}
    for sysn, x in d.groupby("system"):
        g[sysn] = {c: [float(x[c].mean()), float(x[c].std(ddof=1)) if len(x) > 1 else 0.0]
                   for c in ("tavg_recall", "tavg_qps_1thr", "update_tput", "memory_mb", "forced", "sched", "coverage_mean")}
        g[sysn]["n"] = int(len(x))
    S[k] = g
json.dump(S, open(os.path.join(V4, "summary_v4.json"), "w"), indent=1)
NAMES = [("StreamVec", "\\sv"), ("StreamVec-C", "\\sv-C"), ("Skip-C", "\\textsc{Skip-C}"),
         ("Consolidate-0.15", "\\textsc{Consolidate}"), ("Append", "\\textsc{Append}"), ("StreamVec-NoRepair", "\\textsc{NoRepair}")]
cell = lambda g, s: ("%.3f & %.1f & %.1f" % (g[s]["tavg_recall"][0], g[s]["update_tput"][0] / 1e3, g[s]["tavg_qps_1thr"][0] / 1e3)) if s in g else "-- & -- & --"
L = ["\\begin{tabular}{l@{\\hspace{3pt}}rrr@{\\hspace{6pt}}rrr@{\\hspace{6pt}}rrr}", "\\toprule",
     " & \\multicolumn{3}{c}{GloVe, $M{=}16$} & \\multicolumn{3}{c}{SIFT del-heavy, $M{=}16$} & \\multicolumn{3}{c}{SIFT del-heavy, $M{=}8$}\\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
     "System & R@10 & upd. & QPS & R@10 & upd. & QPS & R@10 & upd. & QPS \\\\", "\\midrule"]
for s, nm in NAMES:
    L.append(f"{nm} & {cell(S['glove'], s)} & {cell(S['del16'], s)} & {cell(S['del8'], s)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(TAB, "cpp_general.tex"), "w").write("\n".join(L) + "\n")
for k in S:
    print(k, {s: (round(v["tavg_recall"][0], 4), round(v["update_tput"][0]), round(v["tavg_qps_1thr"][0]), round(v["coverage_mean"][0], 3), round(v["forced"][0]), round(v["sched"][0]), v["n"]) for s, v in S[k].items()})
