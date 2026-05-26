#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-/home/lrh/miniconda3/envs/xunlian/bin/python}"
OUT="results/bdg2_robustness_120"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

echo "[robustness-baselines] waiting for PARBoost to finish: $(date)"
while pgrep -f "scripts/run_par_gbdt_pilot.py --data data/processed/bdg2_electricity_hourly_robustness_120.csv" >/dev/null; do
  sleep 300
done

echo "[robustness-baselines] starting baselines: $(date)"
"$PY" run_forecasting_baselines.py \
  --data data/processed/bdg2_electricity_hourly_robustness_120.csv \
  --output-dir "$OUT/forecasting_baselines" \
  --building-manifest results/bdg2_full_eligible/manifest_active_robustness_120.csv \
  --n-buildings 120 \
  --k 3 7 14 30 \
  --seeds 42 43 44 \
  --models random_forest extra_trees hist_gbdt \
  --max-source-windows 5000 \
  --tree-estimators 80

echo "[robustness-baselines] finished: $(date)"
