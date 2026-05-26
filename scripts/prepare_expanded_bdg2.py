#!/usr/bin/env python
"""Prepare an expanded BDG2 subset without overwriting the existing dataset."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from select_forecasting_buildings import balanced_round_robin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-electricity", default="data/raw/bdg2_electricity.csv")
    ap.add_argument("--metadata", default="data/raw/bdg2_metadata.csv")
    ap.add_argument("--output-data", default="data/processed/bdg2_electricity_hourly_expanded.csv")
    ap.add_argument("--output-manifest", default="results/phase2_paper/expanded/building_manifest_24.csv")
    ap.add_argument("--output-stats", default="data/processed/table_dataset_statistics_expanded.csv")
    ap.add_argument("--n-buildings", type=int, default=24)
    ap.add_argument("--max-per-type", type=int, default=6)
    ap.add_argument("--min-hours", type=int, default=24 * 365)
    args = ap.parse_args()

    wide = pd.read_csv(args.raw_electricity, parse_dates=["timestamp"])
    meta = pd.read_csv(args.metadata)
    meta = meta.set_index("building_id", drop=False)

    building_cols = [c for c in wide.columns if c != "timestamp"]
    stats_rows = []
    for bid in building_cols:
        energy = pd.to_numeric(wide[bid], errors="coerce")
        bmeta = meta.loc[bid] if bid in meta.index else {}
        building_type = bmeta.get("primaryspaceusage", "Unknown") if hasattr(bmeta, "get") else "Unknown"
        sqm = bmeta.get("sqm", np.nan) if hasattr(bmeta, "get") else np.nan
        daily = energy.groupby(wide["timestamp"].dt.hour).mean()
        weekly = energy.groupby(wide["timestamp"].dt.dayofweek).mean()
        stats_rows.append({
            "building_id": bid,
            "building_type": building_type,
            "sqm": sqm,
            "n_hours": int(energy.notna().sum()),
            "start": wide["timestamp"].min().date(),
            "end": wide["timestamp"].max().date(),
            "mean_energy": float(energy.mean()),
            "std_energy": float(energy.std()),
            "cv_energy": float(energy.std() / max(abs(energy.mean()), 1e-8)),
            "daily_profile_range": float(daily.max() - daily.min()),
            "weekly_profile_range": float(weekly.max() - weekly.min()),
            "missing_energy": int(energy.isna().sum()),
        })
    stats = pd.DataFrame(stats_rows)
    stats = stats[stats["n_hours"] >= args.min_hours].copy()

    selected = balanced_round_robin(stats, args.n_buildings, args.max_per_type)
    selected_ids = selected["building_id"].tolist()
    long = wide[["timestamp", *selected_ids]].melt(
        id_vars=["timestamp"], var_name="building_id", value_name="energy"
    )
    long["energy"] = pd.to_numeric(long["energy"], errors="coerce")
    long = long.dropna(subset=["energy"]).sort_values(["building_id", "timestamp"])
    long["hour"] = long["timestamp"].dt.hour
    long["day_of_week"] = long["timestamp"].dt.dayofweek
    long["is_weekend"] = (long["day_of_week"] >= 5).astype(int)
    long["month"] = long["timestamp"].dt.month

    meta_cols = ["building_id", "sqm", "primaryspaceusage"]
    long = long.merge(meta[meta_cols].reset_index(drop=True), on="building_id", how="left")
    long = long.rename(columns={"primaryspaceusage": "building_type"})
    long = long[[
        "timestamp", "building_id", "energy", "hour", "day_of_week",
        "is_weekend", "month", "sqm", "building_type",
    ]]

    Path(args.output_data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_data, index=False)
    selected.to_csv(args.output_manifest, index=False)
    selected.to_csv(args.output_stats, index=False)
    print(selected[["building_id", "building_type", "n_hours", "mean_energy", "cv_energy"]].to_string(index=False))
    print(f"[prepare-expanded] data={args.output_data}")
    print(f"[prepare-expanded] manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
