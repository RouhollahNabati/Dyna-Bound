#!/usr/bin/env bash
# JoS major-revision evaluation: sensitivity → N=100×20 (with DRL) → summarize/Holm/effects.
# Reuses existing N=1000×10 artifacts for CI/effect (no mandatory rerun).
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
LOG="${EVAL}/results/jos_revision_run.log"
mkdir -p "${EVAL}/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) JoS revision start JAVA_HOME=$JAVA_HOME ==="

# Incremental compile once (DRL cold-start overlay change).
(cd "${ROOT}/ifogsim2" && bash compile.sh incremental) || true

echo "=== $(date -Is) sensitivity sweeps ==="
"$PY" "${EVAL}/run_sensitivity.py" --resume --skip-compile

echo "=== $(date -Is) N=100 × 20 journal (GNN/Vector/DCBO/DRL) ==="
# Prefer an already-running parallel ft20 (same out_subdir); only launch if absent.
FT20_DIR="${EVAL}/results/simulator_gnn_lock_s23_ft20"
if [[ ! -d "${FT20_DIR}" ]] || [[ -z "$(ls -A "${FT20_DIR}"/trials_*.csv 2>/dev/null || true)" ]]; then
  "$PY" "${EVAL}/run_grid.py" --grid "${CFG}/grid_s23_ft20.json" --skip-compile --resume
else
  echo "ft20 trial CSVs present — waiting for completion (no second launcher)"
  # Wait until all four policies have 60 rows (3 scenarios × 20 trials) or timeout.
  for i in $(seq 1 720); do
    ready=1
    for pol in colony_gnn colony_rl dcbo drl; do
      f="${FT20_DIR}/trials_${pol}.csv"
      if [[ ! -f "$f" ]]; then ready=0; break; fi
      n=$(tail -n +2 "$f" | wc -l)
      if [[ "$n" -lt 60 ]]; then ready=0; break; fi
    done
    if [[ "$ready" -eq 1 ]]; then
      echo "ft20 complete after wait loop iter=$i"
      break
    fi
    sleep 30
  done
fi
"$PY" "${EVAL}/summarize_smoke.py" \
  --sim "${FT20_DIR}" \
  --out "${EVAL}/results/s23_ft20_summary.csv"
"$PY" "${EVAL}/holm_tests.py" \
  --sim "${FT20_DIR}" \
  --out "${EVAL}/results/holm_s23_ft20.csv"
"$PY" "${EVAL}/effect_sizes.py" \
  --sim "${FT20_DIR}" \
  --out "${EVAL}/results/effect_s23_ft20.csv"

echo "=== $(date -Is) effect sizes on existing N=1000 × 10 ==="
if [[ -d "${EVAL}/results/simulator_gnn_lock_s23_n1000_ft10" ]]; then
  "$PY" "${EVAL}/effect_sizes.py" \
    --sim "${EVAL}/results/simulator_gnn_lock_s23_n1000_ft10" \
    --out "${EVAL}/results/effect_s23_n1000_ft10.csv"
fi

echo "=== $(date -Is) render figures ==="
"$PY" "${EVAL}/render_jos_figures.py"

echo "=== $(date -Is) JoS revision done ==="
