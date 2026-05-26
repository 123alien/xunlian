#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash new_machine_restore_backup.sh ~/xunlian_migration_YYYYMMDD_HHMMSS.tar.gz ~/桌面/xunlian

if [ "$#" -lt 1 ]; then
  echo "Usage: bash new_machine_restore_backup.sh ARCHIVE [TARGET_DIR]"
  exit 1
fi

ARCHIVE="$1"
TARGET_DIR="${2:-${HOME}/桌面/xunlian}"

if [ ! -f "${ARCHIVE}" ]; then
  echo "[restore] archive not found: ${ARCHIVE}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

if [ -f "${ARCHIVE}.sha256" ]; then
  echo "[restore] verifying checksum"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "${ARCHIVE}")" && sha256sum -c "$(basename "${ARCHIVE}").sha256")
  else
    (cd "$(dirname "${ARCHIVE}")" && shasum -a 256 -c "$(basename "${ARCHIVE}").sha256")
  fi
fi

echo "[restore] extracting to ${TARGET_DIR}"
tar -xzf "${ARCHIVE}" -C "${TARGET_DIR}"

cd "${TARGET_DIR}"

echo "[restore] restored project:"
pwd
find . -maxdepth 2 -type d | sort | head -80

echo "[restore] optional environment setup:"
echo "  conda create -n xunlian python=3.10 -y"
echo "  conda activate xunlian"
echo "  pip install -r requirements.txt"

echo "[restore] smoke tests:"
echo "  python -c \"import pandas, numpy, sklearn; print('basic imports ok')\""
echo "  python scripts/make_pa_msr_paper_figures.py"

