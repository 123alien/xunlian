#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/桌面/xunlian/external/iclr2025_dsof}"
ENV_NAME="${ENV_NAME:-dsof}"
GPU_ID="${GPU_ID:-0}"
ITR="${ITR:-2}"

cd "$ROOT"
mkdir -p datasets logs/dsof_queue

source /home/lrh/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "[start] $(date)"
echo "[root] $ROOT"
echo "[env] $ENV_NAME"
echo "[gpu] $GPU_ID"
echo "[itr] $ITR"

download_if_missing() {
  local filename="$1"
  local url="$2"
  if [[ -s "datasets/$filename" ]]; then
    echo "[data] exists $filename $(stat -c%s "datasets/$filename") bytes"
    return 0
  fi
  echo "[data] downloading $filename"
  rm -f "datasets/$filename"
  wget -q --show-progress -O "datasets/$filename.tmp" "$url"
  mv "datasets/$filename.tmp" "datasets/$filename"
  echo "[data] done $filename $(stat -c%s "datasets/$filename") bytes"
}

BASE_URL="https://huggingface.co/datasets/Duyu/Time-Series-Forecasting-Benchmark-Datasets/resolve/main"

download_if_missing "ETTh2.csv" "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
download_if_missing "ETTm1.csv" "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
download_if_missing "Electricity.csv" "$BASE_URL/Electricity.csv"
download_if_missing "Exchange.csv" "$BASE_URL/Exchange.csv"
download_if_missing "Traffic.csv" "$BASE_URL/Traffic.csv"
download_if_missing "Weather.csv" "$BASE_URL/Weather.csv"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local trainer="$3"
  local tag="$4"
  local opt="w_student/DLinear_MLP/${dataset}"

  if [[ ! -f "config/optimizer/${opt}.yaml" ]]; then
    echo "[skip] missing optimizer config/optimizer/${opt}.yaml"
    return 0
  fi

  echo "[run] $(date) dataset=${dataset} pred_len=${pred_len} trainer=${trainer} tag=${tag}"
  python -u src/main.py \
    --itr "$ITR" \
    --pred_len "$pred_len" \
    --data "$dataset" \
    --num_workers 0 \
    --y_model_main DLinear \
    --y_model_student MLP \
    --y_opt "$opt" \
    --y_trainer "$trainer" \
    --comments "$tag"
  echo "[done] $(date) dataset=${dataset} pred_len=${pred_len} trainer=${trainer} tag=${tag}"
}

DATASETS=(Electricity ETTh2 ETTm1 Traffic Exchange Weather)
PRED_LENS=(1 24 48)

for dataset in "${DATASETS[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do
    run_one "$dataset" "$pred_len" "w_student/residual/dsof" "official_dsof_${dataset}_pl${pred_len}"
  done
done

for dataset in "${DATASETS[@]}"; do
  for trainer_name in er td baseline; do
    run_one "$dataset" 1 "w_student/residual/${trainer_name}" "official_${trainer_name}_${dataset}_pl1"
  done
done

echo "[finish] $(date)"
