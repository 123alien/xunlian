#!/usr/bin/env bash
set -u

cd ~/桌面/xunlian || exit 1
mkdir -p logs

PY=/home/lrh/miniconda3/envs/xunlian/bin/python

echo "RESUME_START_NOGATE $(date)"
"$PY" scripts/run_pa_module_extensions.py \
  --data data/processed/cofactor_hourly.csv \
  --output-dir results/pa_module_screen_cofactor_20_nogate \
  --n-buildings 20 \
  --k 3 7 14 30 \
  --seeds 42 \
  --max-source-windows 300 \
  --tree-estimators 100 \
  --top-source-k 5 \
  --variants M1_ANCHOR M2_ANCHOR_SIM M3_ANCHOR_REVIN M5_ANCHOR_REVIN_SIM M7_FULL_REVIN_SIM_CAL
echo "RESUME_NOGATE_STATUS $? $(date)"

echo "RESUME_START_BDG2_MSRDPS $(date)"
"$PY" scripts/run_pa_module_extensions.py \
  --data data/processed/bdg2_electricity_hourly_expanded.csv \
  --output-dir results/pa_module_screen_bdg2_24_msr_dps \
  --n-buildings 24 \
  --k 3 7 14 30 \
  --seeds 42 \
  --max-source-windows 300 \
  --tree-estimators 100 \
  --top-source-k 5 \
  --variants M3_ANCHOR_REVIN M8_ANCHOR_MSR M9_REVIN_MSR M10_REVIN_MSR_DPS M11_REVIN_MSR_DPS_CAL
echo "RESUME_BDG2_STATUS $? $(date)"

echo "RESUME_DONE $(date)"
