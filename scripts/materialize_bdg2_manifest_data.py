#!/usr/bin/env python
"""Materialize a long hourly BDG2 file for a provided building manifest."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-electricity", default="data/raw/bdg2_electricity.csv")
    ap.add_argument("--metadata", default="data/raw/bdg2_metadata.csv")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-data", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    id_col = "building_id" if "building_id" in manifest.columns else "target"
    selected_ids = manifest[id_col].dropna().astype(str).tolist()
    wide = pd.read_csv(args.raw_electricity, parse_dates=["timestamp"], usecols=["timestamp", *selected_ids])
    meta = pd.read_csv(args.metadata)
    meta_cols = ["building_id", "sqm", "primaryspaceusage"]

    long = wide.melt(id_vars=["timestamp"], var_name="building_id", value_name="energy")
    long["energy"] = pd.to_numeric(long["energy"], errors="coerce")
    long = long.dropna(subset=["energy"]).sort_values(["building_id", "timestamp"])
    long["hour"] = long["timestamp"].dt.hour
    long["day_of_week"] = long["timestamp"].dt.dayofweek
    long["is_weekend"] = (long["day_of_week"] >= 5).astype(int)
    long["month"] = long["timestamp"].dt.month
    long = long.merge(meta[meta_cols], on="building_id", how="left")
    long = long.rename(columns={"primaryspaceusage": "building_type"})
    long = long[[
        "timestamp", "building_id", "energy", "hour", "day_of_week",
        "is_weekend", "month", "sqm", "building_type",
    ]]
    out = Path(args.output_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out, index=False)
    print(f"[materialize-bdg2] buildings={len(selected_ids)} rows={len(long)} output={out}")


if __name__ == "__main__":
    main()
