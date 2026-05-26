#!/usr/bin/env python
"""Create ranking, relative metrics, and paired deltas for PARBoost validation runs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_parboost(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"target", "method", "k", "MAE"}.issubset(df.columns):
        return df
    raise ValueError(f"Unexpected PARBoost table columns in {path}: {df.columns.tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parboost-building", required=True)
    ap.add_argument("--baseline-building", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--candidate", default="PAR_HIST_GBDT_SW_CAL")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    par = load_parboost(Path(args.parboost_building))
    base = pd.read_csv(args.baseline_building)
    keep = [
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
        "seasonal_naive_24",
    ]
    base = base[base["method"].isin(keep)]
    cols = ["target", "method", "k", "MAE", "RMSE", "sMAPE", "sigma_err"]
    combo = pd.concat([base[cols], par[cols]], ignore_index=True)
    combo["rank"] = combo.groupby(["target", "k"])["MAE"].rank(method="min")

    piv = combo.pivot_table(index=["target", "k"], columns="method", values="MAE", aggfunc="mean")
    if "persistence" in piv.columns:
        rel = piv.div(piv["persistence"], axis=0).reset_index()
        rel_long = rel.melt(id_vars=["target", "k"], var_name="method", value_name="relative_MAE_vs_persistence")
        combo = combo.merge(rel_long, on=["target", "k", "method"], how="left")
    combo.to_csv(out / "table_validation_building_level.csv", index=False)

    ranking = combo.groupby(["method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        worst_MAE=("MAE", "max"),
        mean_relative_MAE=("relative_MAE_vs_persistence", "mean"),
        median_relative_MAE=("relative_MAE_vs_persistence", "median"),
        mean_rank=("rank", "mean"),
        top1_rate=("rank", lambda x: float((x == 1).mean())),
        top3_rate=("rank", lambda x: float((x <= 3).mean())),
    ).reset_index().sort_values(["k", "mean_rank", "mean_MAE"])
    ranking.to_csv(out / "table_validation_ranking.csv", index=False)

    rows = []
    if args.candidate in piv.columns:
        for baseline in [c for c in keep if c in piv.columns]:
            delta = piv[args.candidate] - piv[baseline]
            ratio = piv[args.candidate] / piv[baseline]
            tmp = pd.DataFrame({
                "delta": delta,
                "ratio": ratio,
            }).reset_index().replace([np.inf, -np.inf], np.nan).dropna()
            for k, g in tmp.groupby("k"):
                rows.append({
                    "candidate": args.candidate,
                    "baseline": baseline,
                    "k": int(k),
                    "mean_delta": float(g["delta"].mean()),
                    "median_delta": float(g["delta"].median()),
                    "worst_delta": float(g["delta"].max()),
                    "best_delta": float(g["delta"].min()),
                    "win_rate": float((g["delta"] < 0).mean()),
                    "mean_ratio": float(g["ratio"].mean()),
                    "median_ratio": float(g["ratio"].median()),
                    "n": int(len(g)),
                })
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(out / "table_validation_pairwise.csv", index=False)
    print(ranking.to_string(index=False))
    if not pairwise.empty:
        print("\nPairwise")
        print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
