"""Experiment 4: scalability in N and D.

Canonical entry point.  The experiment is decomposed into resumable units in
``units.py`` (so each fits the sandbox time limit) and driven to completion by
``run_unit.py``.  This wrapper runs all pending units of exp4 back-to-back.
Run instead, for time-budgeted/resumable execution:
    python run_unit.py exp4 <budget_seconds>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_unit import drive

if __name__ == "__main__":
    # large budget: run every pending unit in one process
    done = drive("exp4", budget=10**9)
    print("exp4:", "complete" if done else "incomplete")
