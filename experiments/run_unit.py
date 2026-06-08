"""Time-budgeted, resumable experiment driver.

Usage:
    python run_unit.py <exp|all> [time_budget_s]

Runs pending units of the chosen experiment(s) until the wall-clock budget is
hit, then exits printing remaining counts.  Re-invoke to resume.
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from experiments.units import REGISTRY


def drive(exp, budget):
    pending_fn, run_fn = REGISTRY[exp]
    t0 = time.perf_counter()
    ran = 0
    while True:
        pend = pending_fn()
        if not pend:
            print(f"{exp}: ALL_DONE")
            return True
        if time.perf_counter() - t0 > budget and ran > 0:
            print(f"{exp}: BUDGET_HIT, {len(pend)} units remaining")
            return False
        u = pend[0]
        ts = time.perf_counter()
        tag = run_fn(u)
        print(f"{exp}: ran {tag} in {time.perf_counter()-ts:.1f}s ({len(pend)-1} remaining)", flush=True)
        ran += 1


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    exps = list(REGISTRY) if which == "all" else [which]
    all_done = True
    for e in exps:
        done = drive(e, budget)
        all_done = all_done and done
        if which == "all" and not done:
            break  # come back next call
    print("OVERALL:", "ALL_DONE" if all_done else "MORE_PENDING")


if __name__ == "__main__":
    main()
