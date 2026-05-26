#!/usr/bin/env bash
set -euo pipefail

# Run this on the old experiment machine from the project root:
#   cd ~/桌面/xunlian
#   bash scripts/remote_make_migration_backup.sh

PROJECT_ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${PROJECT_ROOT}/migration_backups"
ARCHIVE="${BACKUP_DIR}/xunlian_migration_${STAMP}.tar.gz"
MANIFEST="${BACKUP_DIR}/xunlian_migration_${STAMP}_manifest.txt"
SHA256="${ARCHIVE}.sha256"

mkdir -p "${BACKUP_DIR}"

echo "[backup] project root: ${PROJECT_ROOT}"
echo "[backup] archive: ${ARCHIVE}"

INCLUDE_PATHS=(
  "configs"
  "data/processed"
  "docs"
  "figures"
  "scripts"
  "src"
  "requirements.txt"
  "run_augmented_detector.py"
  "run_compare_forecasting_methods.py"
  "run_event_detector.py"
  "run_forecasting_baselines.py"
  "run_forecasting_building_level_stats.py"
  "run_generate_final_forecasting_package.py"
  "run_generate_forecasting_assets.py"
  "run_generate_paper_assets.py"
  "run_m3r_replay.py"
  "run_phase1.py"
  "run_phase2.py"
  "run_scoring_ablation.py"
  "run_statistical_tests.py"
  "run_supplementary.py"
  "results/final_claim_evidence"
  "results/pa_msr_bdg2_120_light_cached_summary"
  "results/pa_module_screen_cofactor_44_msr_dps"
  "results/pa_module_extensions_cofactor_44"
  "results/bdg2_robustness_120"
  "results/pa_srft_pilot_bdg2"
  "results/cofactor_external"
  "results/phase2_paper/expanded_fast"
)

OPTIONAL_PATHS=(
  "external"
  "data/raw"
  "results/phase1"
  "results/phase2"
  "results/par_gbdt_pilot_bdg2"
  "results/cofactor_external/pilot"
)

EXISTING_PATHS=()
MISSING_PATHS=()

for path in "${INCLUDE_PATHS[@]}" "${OPTIONAL_PATHS[@]}"; do
  if [ -e "${path}" ]; then
    EXISTING_PATHS+=("${path}")
  else
    MISSING_PATHS+=("${path}")
  fi
done

{
  echo "Project root: ${PROJECT_ROOT}"
  echo "Created at: ${STAMP}"
  echo
  echo "Included paths:"
  printf '%s\n' "${EXISTING_PATHS[@]}"
  echo
  echo "Missing paths:"
  printf '%s\n' "${MISSING_PATHS[@]}"
} > "${MANIFEST}"

tar -czf "${ARCHIVE}" "${EXISTING_PATHS[@]}"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${ARCHIVE}" > "${SHA256}"
else
  shasum -a 256 "${ARCHIVE}" > "${SHA256}"
fi

echo "[backup] done"
echo "[backup] archive size:"
du -h "${ARCHIVE}"
echo "[backup] manifest: ${MANIFEST}"
echo "[backup] checksum: ${SHA256}"

