#!/usr/bin/env python
"""Summarize the 120-building BDG2 robustness check."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parboost", default="results/bdg2_robustness_120/parboost/aggregate/table_par_gbdt_pilot_building_level.csv")
    ap.add_argument("--baselines", default="results/bdg2_robustness_120/forecasting_baselines/table_forecasting_baselines_building_level.csv")
    ap.add_argument("--output-dir", default="results/bdg2_robustness_120/summary")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    par = pd.read_csv(args.parboost)
    base = pd.read_csv(args.baselines)
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
    combo.to_csv(out / "table_robustness_120_building_level.csv", index=False)
    ranking = combo.groupby(["method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        worst_MAE=("MAE", "max"),
        mean_rank=("rank", "mean"),
        top1_rate=("rank", lambda x: float((x == 1).mean())),
        top3_rate=("rank", lambda x: float((x <= 3).mean())),
    ).reset_index().sort_values(["k", "mean_rank", "mean_MAE"])
    ranking.to_csv(out / "table_robustness_120_ranking.csv", index=False)

    piv = combo.pivot_table(index=["target", "k"], columns="method", values="MAE", aggfunc="mean")
    rows = []
    for base_method in keep:
        if base_method not in piv.columns:
            continue
        delta = piv["PAR_HIST_GBDT_SW_CAL"] - piv[base_method]
        for k, g in delta.reset_index(name="delta").dropna().groupby("k"):
            rows.append({
                "candidate": "PAR_HIST_GBDT_SW_CAL",
                "baseline": base_method,
                "k": int(k),
                "mean_delta": float(g["delta"].mean()),
                "median_delta": float(g["delta"].median()),
                "worst_delta": float(g["delta"].max()),
                "best_delta": float(g["delta"].min()),
                "win_rate": float((g["delta"] < 0).mean()),
                "n": int(len(g)),
            })
    pd.DataFrame(rows).to_csv(out / "table_robustness_120_pairwise.csv", index=False)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
