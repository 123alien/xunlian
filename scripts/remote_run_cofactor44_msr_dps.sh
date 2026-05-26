#!/usr/bin/env bash
set -u

cd ~/桌面/xunlian || exit 1
mkdir -p logs

PY=/home/lrh/miniconda3/envs/xunlian/bin/python

echo "COFACTOR44_MSRDPS_START $(date)"
"$PY" scripts/run_pa_module_extensions.py \
  --data data/processed/cofactor_hourly.csv \
  --output-dir results/pa_module_screen_cofactor_44_msr_dps \
  --n-buildings 44 \
  --k 3 7 14 30 \
  --seeds 42 \
  --max-source-windows 260 \
  --tree-estimators 90 \
  --top-source-k 5 \
  --variants M9_REVIN_MSR M11_REVIN_MSR_DPS_CAL
echo "COFACTOR44_MSRDPS_STATUS $? $(date)"
