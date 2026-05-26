#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON_BIN:-/home/lrh/miniconda3/envs/xunlian/bin/python}"
LOGDIR="results/parboost_postprocess/logs"
mkdir -p "$LOGDIR"

echo "[postprocess-all] waiting for COFACTOR full queue: $(date)"
while pgrep -f "scripts/run_cofactor_full_parboost_queue.sh" >/dev/null || \
      pgrep -f "run_forecasting_baselines.py --data data/processed/cofactor_hourly.csv" >/dev/null || \
      pgrep -f "scripts/run_par_gbdt_pilot.py --data data/processed/cofactor_hourly.csv" >/dev/null; do
  sleep 300
done

echo "[postprocess-all] COFACTOR full finished, summarizing: $(date)"
"$PY" scripts/summarize_parboost_validation.py \
  --parboost-building results/parboost_cofactor_full/parboost/aggregate/table_par_gbdt_pilot_building_level.csv \
  --baseline-building results/parboost_cofactor_full/forecasting_baselines/table_forecasting_baselines_building_level.csv \
  --output-dir results/parboost_cofactor_full/summary \
  > "$LOGDIR/01_cofactor_full_summary.log" 2>&1

echo "[postprocess-all] summarizing BDG2 120 robustness: $(date)"
"$PY" scripts/summarize_parboost_validation.py \
  --parboost-building results/bdg2_robustness_120/parboost/aggregate/table_par_gbdt_pilot_building_level.csv \
  --baseline-building results/bdg2_robustness_120/forecasting_baselines/table_forecasting_baselines_building_level.csv \
  --output-dir results/bdg2_robustness_120/validation_summary \
  > "$LOGDIR/02_bdg2_120_summary.log" 2>&1

echo "[postprocess-all] summarizing BDG2 24 main benchmark: $(date)"
"$PY" scripts/summarize_parboost_validation.py \
  --parboost-building results/parboost_expanded_bdg2/aggregate/table_par_gbdt_pilot_building_level.csv \
  --baseline-building results/phase2_paper/expanded_fast/forecasting_baselines/table_forecasting_baselines_building_level.csv \
  --output-dir results/parboost_expanded_bdg2/validation_summary \
  > "$LOGDIR/03_bdg2_24_summary.log" 2>&1

echo "[postprocess-all] finished: $(date)"
