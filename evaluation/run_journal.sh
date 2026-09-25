#!/usr/bin/env bash
# Journal-scale Dyna-Bound evaluation: N=100×10 then N=1000×10, then summarize + Holm.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$JAVA_HOME/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"

PY="${ROOT}/dyna-bound/offline/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
EVAL="${ROOT}/dyna-bound/evaluation"
CFG="${ROOT}/dyna-bound/configs"
LOG="${EVAL}/results/journal_run.log"
mkdir -p "${EVAL}/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) journal start JAVA_HOME=$JAVA_HOME ==="
# Prefer skip-compile: local jdk-local javac can break without LD_LIBRARY_PATH.
"$PY" "${EVAL}/run_grid.py" --grid "${CFG}/grid_s23_ft10.json" --skip-compile
"$PY" "${EVAL}/summarize_smoke.py" \
  --sim "${EVAL}/results/simulator_gnn_lock_s23_ft10" \
  --out "${EVAL}/results/s23_ft10_summary.csv"
"$PY" "${EVAL}/holm_tests.py" \
  --sim "${EVAL}/results/simulator_gnn_lock_s23_ft10" \
  --out "${EVAL}/results/holm_s23_ft10.csv"

echo "=== $(date -Is) starting N=1000 ==="
"$PY" "${EVAL}/run_grid.py" --grid "${CFG}/grid_s23_n1000_ft10.json" --skip-compile
"$PY" "${EVAL}/summarize_smoke.py" \
  --sim "${EVAL}/results/simulator_gnn_lock_s23_n1000_ft10" \
  --out "${EVAL}/results/s23_n1000_ft10_summary.csv"
"$PY" "${EVAL}/holm_tests.py" \
  --sim "${EVAL}/results/simulator_gnn_lock_s23_n1000_ft10" \
  --out "${EVAL}/results/holm_s23_n1000_ft10.csv"
echo "=== $(date -Is) journal done ==="
