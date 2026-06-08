#!/usr/bin/env bash
# Run the full StreamVec experiment suite.  Single-thread BLAS so parallel
# experiment processes do not oversubscribe the CPU.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results logs

run() {  # run <name> <script>
  echo "[$(date +%H:%M:%S)] START $1"
  python3 "experiments/$2" > "logs/$1.out" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $1 (exit $?)"
}

if [ "${1:-all}" = "all" ]; then
  run exp1 exp1_throughput_recall.py
  run exp2 exp2_recall_vs_tombstone.py
  run exp3 exp3_repair_ablation.py
  run exp4 exp4_scalability.py
  run exp5 exp5_repair_budget.py
  run exp6 exp6_rag_casestudy.py
else
  run "$1" "$2"
fi
echo "[$(date +%H:%M:%S)] ALL EXPERIMENTS COMPLETE"
