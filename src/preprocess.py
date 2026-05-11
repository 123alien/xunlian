"""BDG2 data download, preprocessing, building selection, and feature engineering."""
import os
import json
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

# BDG2 data sources — tried in order until one works
# Zenodo packages the entire repo as a ZIP
BDG2_URLS = [
    "https://zenodo.org/api/records/3887306/files/buds-lab/building-data-genome-project-2-v1.0.zip/content",
]
BDG2_META_URLS = []  # metadata is inside the ZIP


def download_bdg2(raw_dir: str = "data/raw") -> bool:
    """Download BDG2 data files if not already present.

    Returns True if data was downloaded (or already exists).
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)

    data_path = raw / "bdg2_electricity.csv"
    meta_path = raw / "bdg2_metadata.csv"

    if data_path.exists() and meta_path.exists():
        print("[preprocess] BDG2 data already exists — skipping download.")
        return True

    print("[preprocess] Downloading BDG2 repository ZIP from Zenodo (~180 MB)...")
    data_ok = False
    zip_path = raw / "bdg2_repo.zip"
    for url in BDG2_URLS:
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"[preprocess] Downloaded ZIP")
            data_ok = True
            break
        except Exception as e:
            print(f"[preprocess]  Failed: {e}")
            continue

    if not data_ok:
        print("[preprocess] ERROR: Could not download BDG2.")
        print("[preprocess] Please download manually from https://zenodo.org/records/3887306")
        return False

    # Extract ZIP — repository has data/raw/bdg2/electricity.csv and metadata.csv
    print("[preprocess] Extracting ZIP...")
    import shutil
    extract_dir = raw / "_extract"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    os.remove(zip_path)

    # Find electricity and metadata CSV files in extracted tree
    elec_candidates = list(extract_dir.rglob("electricity*.csv"))
    meta_candidates = list(extract_dir.rglob("metadata.csv"))

    if elec_candidates:
        shutil.move(str(elec_candidates[0]), data_path)
        print(f"[preprocess] Found: {elec_candidates[0].name}")
    else:
        all_csvs = list(extract_dir.rglob("*.csv"))
        print(f"Available CSV files: {[f.name for f in all_csvs[:10]]}")
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise RuntimeError("electricity.csv not found in extracted ZIP")

    if meta_candidates:
        shutil.move(str(meta_candidates[0]), meta_path)
        print(f"[preprocess] Found: {meta_candidates[0].name}")

    shutil.rmtree(extract_dir, ignore_errors=True)

    return True


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_raw(raw_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load raw BDG2 data and metadata.

    BDG2 electricity_cleaned.csv is in wide format: timestamp + one column per building.
    Metadata has building_id, primaryspaceusage, sqm, etc.
    """
    raw = Path(raw_dir)
    # Wide-format electricity data: timestamp | building_1 | building_2 | ...
    wide = pd.read_csv(raw / "bdg2_electricity.csv", parse_dates=["timestamp"])
    # Melt to long format
    id_vars = ["timestamp"]
    value_vars = [c for c in wide.columns if c not in id_vars]
    data = wide.melt(id_vars=id_vars, value_vars=value_vars,
                     var_name="building_id", value_name="energy")
    data["energy"] = pd.to_numeric(data["energy"], errors="coerce")

    meta_path = raw / "bdg2_metadata.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    return data, meta


def preprocess(data: pd.DataFrame, meta: pd.DataFrame | None = None,
               min_months: int = 6, max_missing_pct: float = 0.15) -> pd.DataFrame:
    """Clean, filter, and feature-engineer BDG2 data.

    Args:
        data: raw BDG2 dataframe with columns [timestamp, building_id, energy]
        meta: optional metadata with [building_id, sqm, primaryspaceusage]
        min_months: minimum months of data required per building
        max_missing_pct: maximum allowed missing-value percentage

    Returns:
        Cleaned dataframe with columns:
        [timestamp, building_id, energy, hour, day_of_week, is_weekend, month,
         (sqm), (building_type)]
    """
    # Standardise column names
    data = data.copy()
    if "meter_reading" in data.columns and "energy" not in data.columns:
        data = data.rename(columns={"meter_reading": "energy"})

    required = ["timestamp", "building_id", "energy"]
    for col in required:
        if col not in data.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(data.columns)}")

    # Ensure hourly frequency per building
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp").groupby("building_id").resample("h").mean(numeric_only=True)
    data = data.reset_index()  # building_id and timestamp back to columns

    # Filter buildings by data quantity and quality
    stats = data.groupby("building_id").agg(
        n_rows=("energy", "count"),
        missing_pct=("energy", lambda x: x.isna().mean()),
        date_min=("timestamp", "min"),
        date_max=("timestamp", "max"),
    )
    stats["months"] = (stats["date_max"] - stats["date_min"]).dt.days / 30.44
    valid = stats[(stats["months"] >= min_months) & (stats["missing_pct"] <= max_missing_pct)]
    data = data[data["building_id"].isin(valid.index)].copy()

    # Forward-fill short gaps, drop remaining NaN
    data["energy"] = data.groupby("building_id")["energy"].transform(
        lambda x: x.ffill(limit=2)
    )
    data = data.dropna(subset=["energy"])

    # Time features
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["month"] = data["timestamp"].dt.month

    # Metadata merge
    if meta is not None:
        meta_cols = ["building_id"]
        if "sqm" in meta.columns:
            meta_cols.append("sqm")
        if "primaryspaceusage" in meta.columns:
            meta_cols.append("primaryspaceusage")
        if len(meta_cols) > 1:
            data = data.merge(meta[meta_cols], on="building_id", how="left")
            if "primaryspaceusage" in data.columns:
                data = data.rename(columns={"primaryspaceusage": "building_type"})

    # Ensure column order
    out_cols = ["timestamp", "building_id", "energy",
                "hour", "day_of_week", "is_weekend", "month"]
    for c in ["sqm", "building_type"]:
        if c in data.columns:
            out_cols.append(c)

    return data[out_cols].reset_index(drop=True)


def select_buildings(data: pd.DataFrame, n_buildings: int = 5,
                     prefer_types: list[str] | None = None) -> list[str]:
    """Select the N buildings with the longest complete records.

    Prefers diversity in building_type if available.
    """
    stats = data.groupby("building_id").agg(
        n_rows=("energy", "count"),
        missing_pct=("energy", lambda x: x.isna().mean()),
    ).sort_values("n_rows", ascending=False)

    # If building_type available, pick diverse set
    if "building_type" in data.columns and prefer_types:
        chosen = []
        for btype in prefer_types:
            btype_buildings = data[data["building_type"].str.lower() == btype.lower()
                                   ]["building_id"].unique()
            for bid in stats.index:
                if bid in btype_buildings and bid not in chosen:
                    chosen.append(bid)
                    break
        # Fill remainder with longest records
        remaining = [b for b in stats.index if b not in chosen]
        chosen += remaining[:max(0, n_buildings - len(chosen))]
        return chosen[:n_buildings]

    # Fallback: pick top N by record length
    return list(stats.index[:n_buildings])


def make_splits(data: pd.DataFrame, building_id: str,
                train_ratio: float = 0.6, val_ratio: float = 0.2) -> dict:
    """Create chronological train/val/test split for one building.

    Returns dict with keys 'train', 'val', 'test' → DataFrames.
    """
    bldg = data[data["building_id"] == building_id].sort_values("timestamp").copy()
    n = len(bldg)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": bldg.iloc[:train_end],
        "val": bldg.iloc[train_end:val_end],
        "test": bldg.iloc[val_end:],
    }


def save_dataset_stats(data: pd.DataFrame, path: str):
    """Save per-building statistics table."""
    stats = data.groupby("building_id").agg(
        n_rows=("energy", "count"),
        missing_pct=("energy", lambda x: x.isna().mean()),
        date_start=("timestamp", "min"),
        date_end=("timestamp", "max"),
        mean_kwh=("energy", "mean"),
        std_kwh=("energy", "std"),
    )
    if "sqm" in data.columns:
        stats["sqm"] = data.groupby("building_id")["sqm"].first()
    if "building_type" in data.columns:
        stats["building_type"] = data.groupby("building_id")["building_type"].first()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path)
    print(f"[preprocess] Dataset statistics saved to {path}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(raw_dir: str = "data/raw",
                                processed_dir: str = "data/processed",
                                n_buildings: int = 5) -> pd.DataFrame:
    """Full preprocessing pipeline: download → clean → select buildings → save.

    Returns the cleaned dataframe.
    """
    # Download
    ok = download_bdg2(raw_dir)
    if not ok:
        raise RuntimeError("BDG2 download failed. See messages above.")

    # Load
    data, meta = load_raw(raw_dir)
    print(f"[preprocess] Loaded {len(data):,} rows, {data['building_id'].nunique()} buildings.")

    # Preprocess
    data = preprocess(data, meta)
    print(f"[preprocess] After cleaning: {len(data):,} rows, {data['building_id'].nunique()} buildings.")

    # Select buildings
    prefer = ["office", "education", "lodging"] if "building_type" in data.columns else None
    selected = select_buildings(data, n_buildings, prefer)
    data = data[data["building_id"].isin(selected)].copy()
    print(f"[preprocess] Selected {len(selected)} buildings: {selected}")

    # Save
    out_path = Path(processed_dir) / "bdg2_electricity_hourly.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"[preprocess] Processed data saved to {out_path}")

    save_dataset_stats(data, str(Path(processed_dir) / "table_dataset_statistics.csv"))

    return data
