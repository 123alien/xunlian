#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-/home/lrh/miniconda3/envs/xunlian/bin/python}"
OUT="results/parboost_cofactor_full"
DATA="data/processed/cofactor_hourly.csv"
MANIFEST="results/cofactor_external/building_manifest_active.csv"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

echo "[cofactor-full] started $(date)"

"$PY" run_forecasting_baselines.py \
  --data "$DATA" \
  --output-dir "$OUT/forecasting_baselines" \
  --building-manifest "$MANIFEST" \
  --n-buildings 44 \
  --k 3 7 14 30 \
  --seeds 42 43 44 \
  --models random_forest extra_trees hist_gbdt \
  --max-source-windows 5000 \
  --tree-estimators 80 \
  > "$LOGDIR/01_baselines.log" 2>&1

"$PY" scripts/run_par_gbdt_pilot.py \
  --data "$DATA" \
  --building-manifest "$MANIFEST" \
  --output-dir "$OUT/parboost" \
  --n-buildings 44 \
  --k 3 7 14 30 \
  --seeds 42 43 44 \
  --models par_hist_gbdt_sw_cal par_extra_trees_sw_cal \
  --max-source-windows 1200 \
  --target-weight 25 \
  --tree-estimators 240 \
  --top-source-k 5 \
  > "$LOGDIR/02_parboost.log" 2>&1

echo "[cofactor-full] finished $(date)"
