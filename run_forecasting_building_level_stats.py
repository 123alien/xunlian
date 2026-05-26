#!/usr/bin/env python
"""Building-level statistics for the forecasting-focused paper.

Seeds are repeated runs, not independent buildings. This script first averages
M3-M2 deltas across seeds within each target building, then performs paired
tests over buildings (n=12 by default). Use these tables for main-text claims.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_statistical_tests import bootstrap_ci_mean, cliffs_delta, wilcoxon_signed_rank


METRICS = ["MAE", "RMSE", "sMAPE", "sigma_err", "sigma_score"]


def paired_seed_level(few: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = few[few["method"].isin(["M2_target_only", "M3_pretrain_ft"])].copy()
    for key, g in subset.groupby(["target", "model", "k", "seed"]):
        m2 = g[g["method"] == "M2_target_only"]
        m3 = g[g["method"] == "M3_pretrain_ft"]
        if len(m2) == 0 or len(m3) == 0:
            continue
        m2 = m2.iloc[0]
        m3 = m3.iloc[0]
        row = {"target": key[0], "model": key[1], "k": int(key[2]), "seed": int(key[3])}
        for metric in METRICS:
            row[f"delta_{metric}"] = float(m3[metric] - m2[metric])
            row[f"M2_{metric}"] = float(m2[metric])
            row[f"M3_{metric}"] = float(m3[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_building_level(seed_pairs: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in seed_pairs.columns if c.startswith(("delta_", "M2_", "M3_"))]
    return (
        seed_pairs
        .groupby(["target", "model", "k"], as_index=False)[value_cols]
        .mean()
    )


def summarize_claims(building_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, k), g in building_pairs.groupby(["model", "k"]):
        row = {"model": model, "k": int(k), "n_buildings": int(g["target"].nunique())}
        for metric in METRICS:
            vals = g[f"delta_{metric}"].to_numpy(float)
            ci_low, ci_high = bootstrap_ci_mean(vals, n_boot=10000, seed=42)
            w, p = wilcoxon_signed_rank(vals)
            row[f"{metric}_mean_delta"] = float(np.mean(vals))
            row[f"{metric}_median_delta"] = float(np.median(vals))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
            row[f"{metric}_wilcoxon_p"] = p
            row[f"{metric}_cliffs_delta"] = cliffs_delta(vals)
            row[f"{metric}_win_rate"] = float(np.mean(vals < 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--output-dir", default="results/phase2_paper/forecasting_paper")
    args = ap.parse_args()

    agg = Path(args.aggregate_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    few = pd.read_csv(agg / "table_few_shot_transfer_all_models.csv")
    seed_pairs = paired_seed_level(few)
    building_pairs = aggregate_building_level(seed_pairs)
    claim = summarize_claims(building_pairs)

    seed_pairs.to_csv(out / "table_seed_level_m2_m3_deltas.csv", index=False)
    building_pairs.to_csv(out / "table_building_level_m2_m3_deltas.csv", index=False)
    claim.to_csv(out / "table_building_level_claim_summary.csv", index=False)

    print(claim.to_string(index=False))
    print(f"[forecasting-stats] saved to {out}")


if __name__ == "__main__":
    main()
