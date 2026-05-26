#!/usr/bin/env python
"""Compare neural transfer methods against classical forecasting baselines."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--baseline-dir", default="results/phase2_paper/forecasting_baselines")
    ap.add_argument("--output-dir", default="results/phase2_paper/forecasting_paper")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7, 14])
    args = ap.parse_args()

    agg = Path(args.aggregate_dir)
    base = Path(args.baseline_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    few_path = agg / "table_few_shot_transfer_all_models.csv"
    if not few_path.exists():
        few_path = agg / "table_few_shot_transfer.csv"
    few = pd.read_csv(few_path)
    few = few[few["method"].isin(["M2_target_only", "M3_pretrain_ft"]) & few["k"].isin(args.k)].copy()
    neural = few[["target", "k", "seed", "MAE", "RMSE", "sMAPE", "sigma_err"]].copy()
    neural["method"] = few["model"] + "_" + few["method"]
    neural = neural[["target", "k", "seed", "method", "MAE", "RMSE", "sMAPE", "sigma_err"]]

    baselines = pd.read_csv(base / "table_forecasting_baselines.csv")
    baselines = baselines[["target", "k", "seed", "method", "MAE", "RMSE", "sMAPE", "sigma_err"]]

    m3r_path = agg / "table_m3r_replay.csv"
    extra = []
    if m3r_path.exists():
        m3r = pd.read_csv(m3r_path)
        m3r = m3r[m3r["k"].isin(args.k)].copy()
        m3r["method"] = (
            m3r["model"] + "_M3R_lambda" + m3r["lambda_source"].map(lambda x: f"{float(x):g}")
        )
        extra.append(m3r[["target", "k", "seed", "method", "MAE", "RMSE", "sMAPE", "sigma_err"]])

    combined = pd.concat([neural, baselines, *extra], ignore_index=True)
    combined.to_csv(out / "table_all_forecasting_methods.csv", index=False)

    summary = combined.groupby(["method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
    summary.to_csv(out / "table_all_forecasting_methods_summary.csv", index=False)

    # Building-level leaderboard: average seeds first, then rank methods per target/k.
    bld = combined.groupby(["target", "k", "method"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
    bld["rank_MAE"] = bld.groupby(["target", "k"])["MAE"].rank(method="min", ascending=True)
    bld.to_csv(out / "table_all_forecasting_methods_building_level.csv", index=False)

    leader = bld.groupby(["method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_sigma_err=("sigma_err", "mean"),
        mean_rank_MAE=("rank_MAE", "mean"),
        top1_rate=("rank_MAE", lambda x: float(np.mean(x == 1))),
        top3_rate=("rank_MAE", lambda x: float(np.mean(x <= 3))),
    ).reset_index().sort_values(["k", "mean_rank_MAE", "mean_MAE"])
    leader.to_csv(out / "table_forecasting_leaderboard.csv", index=False)
    print(leader.to_string(index=False))
    print(f"[compare-forecasting] saved to {out}")


if __name__ == "__main__":
    main()
