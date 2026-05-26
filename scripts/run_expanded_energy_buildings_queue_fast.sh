#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-/home/lrh/miniconda3/envs/xunlian/bin/python}"
OUT="results/phase2_paper/expanded_fast"
DATA="data/processed/bdg2_electricity_hourly_expanded.csv"
MANIFEST="$OUT/building_manifest_24.csv"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

echo "[queue-fast] started $(date)"
"$PY" scripts/prepare_expanded_bdg2.py \
  --output-data "$DATA" \
  --output-manifest "$MANIFEST" \
  --output-stats data/processed/table_dataset_statistics_expanded.csv \
  --n-buildings 24 \
  --max-per-type 6 \
  > "$LOGDIR/00_prepare_expanded.log" 2>&1

"$PY" -u scripts/run_neural_transfer_fast.py \
  --data "$DATA" \
  --output-dir "$OUT/neural" \
  --n-buildings 24 \
  --building-manifest "$MANIFEST" \
  --k 0 3 7 14 30 \
  --seeds 42 43 44 45 46 \
  --models lstm dlinear patchtst \
  > "$LOGDIR/01_neural_fast.log" 2>&1

"$PY" -u run_forecasting_baselines.py \
  --data "$DATA" \
  --output-dir "$OUT/forecasting_baselines" \
  --n-buildings 24 \
  --building-manifest "$MANIFEST" \
  --k 3 7 14 30 \
  --seeds 42 43 44 45 46 \
  --models ridge random_forest extra_trees hist_gbdt \
  --max-source-windows 10000 \
  --tree-estimators 160 \
  > "$LOGDIR/02_classical_baselines.log" 2>&1

"$PY" -u run_m3r_replay.py \
  --data "$DATA" \
  --phase2-dir "$OUT/neural" \
  --output-dir "$OUT/neural" \
  --n-buildings 24 \
  --building-manifest "$MANIFEST" \
  --models dlinear \
  --k 3 7 14 30 \
  --seeds 42 43 44 45 46 \
  --lambda-source 0.05 0.1 0.2 0.3 \
  > "$LOGDIR/03_dlinear_srft_lambda_ablation.log" 2>&1

"$PY" -u run_compare_forecasting_methods.py \
  --aggregate-dir "$OUT/neural/aggregate" \
  --baseline-dir "$OUT/forecasting_baselines" \
  --output-dir "$OUT/forecasting_paper" \
  --k 3 7 14 30 \
  > "$LOGDIR/04_compare_methods.log" 2>&1

"$PY" -u scripts/source_target_similarity_analysis.py \
  --data "$DATA" \
  --building-manifest "$MANIFEST" \
  --method-table "$OUT/forecasting_paper/table_all_forecasting_methods.csv" \
  --output-dir "$OUT/similarity" \
  --model dlinear \
  > "$LOGDIR/05_similarity.log" 2>&1

echo "[queue-fast] finished $(date)"
