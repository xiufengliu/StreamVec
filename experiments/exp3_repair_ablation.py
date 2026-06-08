"""Experiment 3: repair-scheduling ablation.

Canonical entry point.  The experiment is decomposed into resumable units in
``units.py`` (so each fits the sandbox time limit) and driven to completion by
``run_unit.py``.  This wrapper runs all pending units of exp3 back-to-back.
Run instead, for time-budgeted/resumable execution:
    python run_unit.py exp3 <budget_seconds>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_unit import drive

if __name__ == "__main__":
    # large budget: run every pending unit in one process
    done = drive("exp3", budget=10**9)
    print("exp3:", "complete" if done else "incomplete")
