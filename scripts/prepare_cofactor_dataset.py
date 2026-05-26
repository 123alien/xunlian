#!/usr/bin/env python
"""Prepare COFACTOR Drammen buildings into the project forecasting format."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_building_file(path: Path) -> tuple[pd.DataFrame, dict]:
    lines = path.read_text(errors="replace").splitlines()
    header_line = int(lines[0].split(";")[1])
    meta = {}
    for line in lines[1:header_line - 1]:
        if not line or ";" not in line:
            continue
        key, value = line.split(";", 1)
        meta[key] = value
    df = pd.read_csv(path, sep=";", skiprows=header_line - 1)
    df["timestamp"] = pd.to_datetime(df["TimeStamp"], utc=True).dt.tz_convert(None)
    df["building_id"] = "cofactor_" + str(meta.get("building_id", path.stem.replace("building_", "")))
    df["energy"] = pd.to_numeric(df["ElImp"], errors="coerce") / 1000.0
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["sqm"] = pd.to_numeric(meta.get("floor_area", np.nan), errors="coerce")
    df["building_type"] = meta.get("building_category", "unknown")
    for raw, out in [("Tout", "temp_outdoor"), ("SolGlob", "solar_global"), ("WindSpd", "wind_speed"), ("WindDir", "wind_dir")]:
        if raw in df.columns:
            df[out] = pd.to_numeric(df[raw], errors="coerce")
    keep = [
        "timestamp", "building_id", "energy", "hour", "day_of_week", "is_weekend",
        "month", "sqm", "building_type", "temp_outdoor", "solar_global", "wind_speed", "wind_dir",
    ]
    return df[[c for c in keep if c in df.columns]], meta


def test_profile_stats(g: pd.DataFrame) -> dict:
    g = g.sort_values("timestamp")
    n = len(g)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train = g.iloc[:train_end]
    test = g.iloc[val_end:]
    energy = test["energy"].to_numpy(dtype=float)
    diff = np.abs(np.diff(energy))
    return {
        "n_hours": n,
        "start": g["timestamp"].min(),
        "end": g["timestamp"].max(),
        "train_mean": float(train["energy"].mean()),
        "train_std": float(train["energy"].std()),
        "test_mean": float(np.mean(energy)),
        "test_std": float(np.std(energy)),
        "test_min": float(np.min(energy)),
        "test_max": float(np.max(energy)),
        "test_zero_ratio": float(np.mean(np.isclose(energy, 0.0))),
        "test_diff_mean": float(np.mean(diff)) if len(diff) else np.nan,
        "test_diff_p95": float(np.percentile(diff, 95)) if len(diff) else np.nan,
        "missing_energy": int(g["energy"].isna().sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/cofactor/extracted/COFACTOR_Drammen_Buildings")
    ap.add_argument("--output-data", default="data/processed/cofactor_hourly.csv")
    ap.add_argument("--output-stats", default="data/processed/cofactor_building_stats.csv")
    ap.add_argument("--active-manifest", default="results/cofactor_external/building_manifest_active.csv")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    frames = []
    meta_rows = []
    for path in sorted(raw.glob("building_*.txt")):
        df, meta = parse_building_file(path)
        frames.append(df)
        meta_rows.append(meta)
    data = pd.concat(frames, ignore_index=True).sort_values(["building_id", "timestamp"])
    data = data.dropna(subset=["energy"])

    stats_rows = []
    for bid, g in data.groupby("building_id"):
        row = {"building_id": bid}
        row.update(test_profile_stats(g))
        row["building_type"] = g["building_type"].iloc[0]
        row["sqm"] = g["sqm"].iloc[0] if "sqm" in g else np.nan
        row["mean_energy"] = float(g["energy"].mean())
        row["std_energy"] = float(g["energy"].std())
        row["cv_energy"] = row["std_energy"] / max(abs(row["mean_energy"]), 1e-8)
        daily = g.groupby(g["timestamp"].dt.hour)["energy"].mean()
        weekly = g.groupby(g["timestamp"].dt.dayofweek)["energy"].mean()
        row["daily_profile_range"] = float(daily.max() - daily.min())
        row["weekly_profile_range"] = float(weekly.max() - weekly.min())
        row["active_primary"] = (
            row["test_zero_ratio"] < 0.5
            and row["test_mean"] > 1e-3
            and row["test_std"] > 1e-3
            and row["n_hours"] >= 24 * 365
        )
        stats_rows.append(row)
    stats = pd.DataFrame(stats_rows).sort_values(["active_primary", "building_type", "building_id"], ascending=[False, True, True])

    Path(args.output_data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.active_manifest).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_data, index=False)
    stats.to_csv(args.output_stats, index=False)
    stats[stats["active_primary"]].to_csv(args.active_manifest, index=False)
    print(f"[cofactor] buildings={data['building_id'].nunique()} rows={len(data)}")
    print(f"[cofactor] active={int(stats['active_primary'].sum())}")
    print(stats[["building_id", "building_type", "n_hours", "test_mean", "test_std", "test_zero_ratio", "active_primary"]].to_string(index=False))
    print(f"[cofactor] data={args.output_data}")
    print(f"[cofactor] active_manifest={args.active_manifest}")


if __name__ == "__main__":
    main()
