#!/usr/bin/env python
"""Select a balanced building manifest for expanded forecasting experiments."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_buildings(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bid, g in data.groupby("building_id"):
        g = g.sort_values("timestamp")
        energy = g["energy"].astype(float)
        building_type = str(g["building_type"].dropna().iloc[0]) if "building_type" in g else "Unknown"
        sqm = float(g["sqm"].dropna().iloc[0]) if "sqm" in g and g["sqm"].notna().any() else np.nan
        daily = energy.groupby(g["timestamp"].dt.hour).mean()
        weekly = energy.groupby(g["timestamp"].dt.dayofweek).mean()
        rows.append({
            "building_id": bid,
            "building_type": building_type,
            "sqm": sqm,
            "n_hours": len(g),
            "start": g["timestamp"].min().date(),
            "end": g["timestamp"].max().date(),
            "mean_energy": energy.mean(),
            "std_energy": energy.std(),
            "cv_energy": energy.std() / max(abs(energy.mean()), 1e-8),
            "daily_profile_range": daily.max() - daily.min(),
            "weekly_profile_range": weekly.max() - weekly.min(),
            "missing_energy": int(energy.isna().sum()),
        })
    return pd.DataFrame(rows)


def balanced_round_robin(stats: pd.DataFrame, n_buildings: int, max_per_type: int) -> pd.DataFrame:
    eligible = stats[(stats["missing_energy"] == 0) & (stats["n_hours"] >= 24 * 365)].copy()
    eligible["type_count"] = eligible.groupby("building_type")["building_id"].transform("count")
    eligible = eligible.sort_values(
        ["type_count", "building_type", "n_hours", "cv_energy"],
        ascending=[True, True, False, False],
    )

    buckets = {
        typ: grp.sort_values(["cv_energy", "mean_energy"], ascending=[False, False]).to_dict("records")
        for typ, grp in eligible.groupby("building_type", sort=True)
    }
    selected = []
    per_type = {typ: 0 for typ in buckets}
    while len(selected) < n_buildings:
        progressed = False
        for typ in sorted(buckets, key=lambda t: (per_type[t], len(buckets[t]), t)):
            if len(selected) >= n_buildings:
                break
            if per_type[typ] >= max_per_type or not buckets[typ]:
                continue
            selected.append(buckets[typ].pop(0))
            per_type[typ] += 1
            progressed = True
        if not progressed:
            break
    return pd.DataFrame(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    ap.add_argument("--output", default="results/phase2_paper/expanded/building_manifest_24.csv")
    ap.add_argument("--n-buildings", type=int, default=24)
    ap.add_argument("--max-per-type", type=int, default=6)
    args = ap.parse_args()

    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    stats = summarize_buildings(data)
    selected = balanced_round_robin(stats, args.n_buildings, args.max_per_type)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False)
    stats.to_csv(out.with_name(out.stem + "_all_candidates.csv"), index=False)
    print(selected[["building_id", "building_type", "n_hours", "mean_energy", "cv_energy"]].to_string(index=False))
    print(f"[select-buildings] selected={len(selected)} saved={out}")


if __name__ == "__main__":
    main()
