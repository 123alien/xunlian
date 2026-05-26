#!/usr/bin/env python
"""Prepare a BDG2 eligible/active subset for supplementary robustness checks."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def persistence_mae(values: np.ndarray, window_size: int = 24, horizon: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) <= window_size + horizon:
        return float("nan")
    y_true = values[window_size + horizon - 1:]
    y_pred = values[window_size - 1: len(values) - horizon]
    n = min(len(y_true), len(y_pred))
    if n <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[:n] - y_pred[:n])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-electricity", default="data/raw/bdg2_electricity.csv")
    ap.add_argument("--metadata", default="data/raw/bdg2_metadata.csv")
    ap.add_argument("--output-data", default="data/processed/bdg2_electricity_hourly_full_eligible_active.csv")
    ap.add_argument("--output-manifest", default="results/bdg2_full_eligible/manifest_active.csv")
    ap.add_argument("--output-audit", default="results/bdg2_full_eligible/table_eligibility_audit.csv")
    ap.add_argument("--min-hours", type=int, default=24 * 300)
    ap.add_argument("--max-missing-rate", type=float, default=0.2)
    ap.add_argument("--min-train-hours", type=int, default=24 * 30)
    ap.add_argument("--min-val-hours", type=int, default=24 * 14)
    ap.add_argument("--min-test-hours", type=int, default=24 * 14)
    ap.add_argument("--min-test-std", type=float, default=1e-3)
    ap.add_argument("--min-test-persistence-mae", type=float, default=1e-3)
    ap.add_argument("--max-zero-ratio", type=float, default=0.95)
    ap.add_argument("--max-buildings", type=int, default=0, help="0 keeps all eligible active buildings.")
    args = ap.parse_args()

    wide = pd.read_csv(args.raw_electricity, parse_dates=["timestamp"])
    meta = pd.read_csv(args.metadata)
    meta = meta.set_index("building_id", drop=False)
    total_rows = len(wide)
    building_cols = [c for c in wide.columns if c != "timestamp"]

    rows = []
    for bid in building_cols:
        s = pd.to_numeric(wide[bid], errors="coerce")
        valid = s.dropna()
        missing_rate = 1.0 - len(valid) / max(total_rows, 1)
        bmeta = meta.loc[bid] if bid in meta.index else {}
        building_type = bmeta.get("primaryspaceusage", "Unknown") if hasattr(bmeta, "get") else "Unknown"
        sqm = bmeta.get("sqm", np.nan) if hasattr(bmeta, "get") else np.nan
        g = pd.DataFrame({"timestamp": wide.loc[s.notna(), "timestamp"], "energy": valid.to_numpy()})
        n = len(g)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        train = g.iloc[:train_end]
        val = g.iloc[train_end:val_end]
        test = g.iloc[val_end:]
        e = g["energy"].to_numpy(dtype=float)
        test_e = test["energy"].to_numpy(dtype=float)
        daily = g["energy"].groupby(g["timestamp"].dt.hour).mean() if n else pd.Series(dtype=float)
        rows.append({
            "building_id": bid,
            "building_type": building_type,
            "sqm": sqm,
            "n_hours": int(n),
            "missing_rate": float(missing_rate),
            "train_hours": int(len(train)),
            "val_hours": int(len(val)),
            "test_hours": int(len(test)),
            "mean_energy": float(np.mean(e)) if n else np.nan,
            "std_energy": float(np.std(e)) if n else np.nan,
            "zero_ratio": float(np.mean(np.isclose(e, 0.0))) if n else np.nan,
            "test_mean": float(np.mean(test_e)) if len(test_e) else np.nan,
            "test_std": float(np.std(test_e)) if len(test_e) else np.nan,
            "test_zero_ratio": float(np.mean(np.isclose(test_e, 0.0))) if len(test_e) else np.nan,
            "test_persistence_mae": persistence_mae(test_e),
            "daily_profile_range": float(daily.max() - daily.min()) if len(daily) else np.nan,
        })

    audit = pd.DataFrame(rows)
    audit["eligible_coverage"] = (
        (audit["n_hours"] >= args.min_hours)
        & (audit["missing_rate"] <= args.max_missing_rate)
        & (audit["train_hours"] >= args.min_train_hours)
        & (audit["val_hours"] >= args.min_val_hours)
        & (audit["test_hours"] >= args.min_test_hours)
    )
    audit["active_test"] = (
        (audit["test_std"] > args.min_test_std)
        & (audit["test_persistence_mae"] > args.min_test_persistence_mae)
        & (audit["test_zero_ratio"] < args.max_zero_ratio)
    )
    audit["selected"] = audit["eligible_coverage"] & audit["active_test"]
    selected = audit[audit["selected"]].sort_values(
        ["building_type", "test_persistence_mae", "building_id"],
        ascending=[True, False, True],
    )
    if args.max_buildings and args.max_buildings > 0:
        selected = selected.head(args.max_buildings)

    selected_ids = selected["building_id"].tolist()
    if not selected_ids:
        raise SystemExit("No eligible active buildings selected.")

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
    Path(args.output_audit).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output_data, index=False)
    selected.to_csv(args.output_manifest, index=False)
    audit.to_csv(args.output_audit, index=False)
    print(f"[bdg2-full-eligible] raw_buildings={len(audit)}")
    print(f"[bdg2-full-eligible] coverage_eligible={int(audit['eligible_coverage'].sum())}")
    print(f"[bdg2-full-eligible] active_selected={len(selected)}")
    print(selected[["building_id", "building_type", "n_hours", "test_std", "test_persistence_mae", "test_zero_ratio"]].to_string(index=False))
    print(f"[bdg2-full-eligible] data={args.output_data}")
    print(f"[bdg2-full-eligible] manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
