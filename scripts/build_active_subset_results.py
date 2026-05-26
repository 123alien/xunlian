#!/usr/bin/env python
"""Build active-test subset summaries from expanded forecasting results."""
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("results/phase2_paper/expanded_fast")
DIAG = BASE / "diagnostics"
OUT = BASE / "active_subset"


def summarize_leaderboard(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = df.groupby(["method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
    bld = df.groupby(["target", "k", "method"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
    bld["rank_MAE"] = bld.groupby(["target", "k"])["MAE"].rank(method="min", ascending=True)
    leader = bld.groupby(["method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_sigma_err=("sigma_err", "mean"),
        mean_rank_MAE=("rank_MAE", "mean"),
        top1_rate=("rank_MAE", lambda x: float(np.mean(x == 1))),
        top3_rate=("rank_MAE", lambda x: float(np.mean(x <= 3))),
    ).reset_index().sort_values(["k", "mean_rank_MAE", "mean_MAE"])
    return summary, leader


def pairwise(df: pd.DataFrame, a: str, b: str, out_col: str) -> pd.DataFrame:
    bld = df.groupby(["target", "k", "method"])[["MAE", "sigma_err"]].mean().reset_index()
    piv = bld[bld["method"].isin([a, b])].pivot_table(index=["target", "k"], columns="method", values="MAE")
    if a not in piv or b not in piv:
        return pd.DataFrame()
    delta = piv[b] - piv[a]
    return delta.reset_index(name=out_col).groupby("k")[out_col].agg(
        mean_delta="mean",
        median_delta="median",
        win_rate=lambda x: float((x < 0).mean()),
        n="count",
    ).reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(DIAG / "table_data_script_model_audit.csv")
    allm = pd.read_csv(BASE / "forecasting_paper/table_all_forecasting_methods.csv")
    bld = pd.read_csv(BASE / "forecasting_paper/table_all_forecasting_methods_building_level.csv")

    active = audit[
        (audit["test_zero_ratio"] < 0.5)
        & (audit["test_std"] > 1e-3)
        & (audit["test_mean"] > 1e-3)
        & (audit["manual_persistence_mae"] > 1e-3)
    ].copy()
    inactive = audit[~audit["target"].isin(active["target"])].copy()

    active_ids = sorted(active["target"].tolist())
    inactive_ids = sorted(inactive["target"].tolist())
    pd.DataFrame({"target": active_ids}).to_csv(OUT / "active_buildings.csv", index=False)
    pd.DataFrame({"target": inactive_ids}).to_csv(OUT / "inactive_or_near_constant_buildings.csv", index=False)
    active.to_csv(OUT / "table_active_building_audit.csv", index=False)
    inactive.to_csv(OUT / "table_inactive_building_audit.csv", index=False)

    active_all = allm[allm["target"].isin(active_ids)].copy()
    active_bld = bld[bld["target"].isin(active_ids)].copy()
    active_all.to_csv(OUT / "table_all_forecasting_methods_active.csv", index=False)
    active_bld.to_csv(OUT / "table_all_forecasting_methods_building_level_active.csv", index=False)

    summary, leader = summarize_leaderboard(active_all)
    summary.to_csv(OUT / "table_method_summary_active.csv", index=False)
    leader.to_csv(OUT / "table_leaderboard_active.csv", index=False)

    pairs = []
    for model in ["lstm", "dlinear", "patchtst"]:
        res = pairwise(active_all, f"{model}_M2_target_only", f"{model}_M3_pretrain_ft", f"{model}_M3_minus_M2")
        if not res.empty:
            res.insert(0, "comparison", f"{model}_M3_vs_M2")
            pairs.append(res)
    for lam in ["0.05", "0.1", "0.2", "0.3"]:
        res = pairwise(active_all, "dlinear_M3_pretrain_ft", f"dlinear_M3R_lambda{lam}", f"dlinear_M3R_lambda{lam}_minus_M3")
        if not res.empty:
            res.insert(0, "comparison", f"dlinear_M3R_lambda{lam}_vs_M3")
            pairs.append(res)
    pair_df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    pair_df.to_csv(OUT / "table_pairwise_claims_active.csv", index=False)

    # Add normalized MAE by building mean.
    feat = pd.read_csv(BASE / "similarity/table_building_similarity_features.csv").rename(columns={"building_id": "target"})
    active_bld_meta = active_bld.merge(feat[["target", "building_type", "mean_energy"]], on="target", how="left")
    active_bld_meta["nMAE_pct"] = active_bld_meta["MAE"] / active_bld_meta["mean_energy"] * 100
    active_bld_meta.to_csv(OUT / "table_building_level_active_with_nmae.csv", index=False)
    nmae = active_bld_meta.groupby(["method", "k"]).agg(
        mean_nMAE_pct=("nMAE_pct", "mean"),
        median_nMAE_pct=("nMAE_pct", "median"),
    ).reset_index().sort_values(["k", "median_nMAE_pct"])
    nmae.to_csv(OUT / "table_nmae_summary_active.csv", index=False)

    type_summary = active_bld_meta.groupby(["building_type", "k", "method"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_nMAE_pct=("nMAE_pct", "mean"),
        median_nMAE_pct=("nMAE_pct", "median"),
        mean_rank=("rank_MAE", "mean"),
        n_buildings=("target", "nunique"),
    ).reset_index()
    type_summary.to_csv(OUT / "table_building_type_summary_active.csv", index=False)

    report = []
    report.append("# Active Subset Result Summary\n")
    report.append(f"Active buildings: {len(active_ids)}")
    report.append(f"Inactive/near-constant buildings: {len(inactive_ids)}\n")
    report.append("## Active building IDs\n")
    report.append("\n".join(active_ids))
    report.append("\n## Inactive/near-constant building IDs\n")
    report.append("\n".join(inactive_ids))
    report.append("\n## Active leaderboard top 10 by k\n")
    cols = ["method", "mean_MAE", "median_MAE", "mean_rank_MAE", "top1_rate", "top3_rate"]
    for k in sorted(leader["k"].unique()):
        report.append(f"\n### k={k}\n")
        report.append(leader[leader["k"] == k][cols].head(10).to_string(index=False))
    report.append("\n## Active pairwise claims\n")
    report.append(pair_df.to_string(index=False))
    (OUT / "active_subset_summary.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[active-subset] active={len(active_ids)} inactive={len(inactive_ids)} saved={OUT}")
    print("\nActive IDs:", active_ids)
    print("\nInactive IDs:", inactive_ids)
    print("\nLeaderboard top 8:")
    for k in sorted(leader["k"].unique()):
        print(f"\nK={k}")
        print(leader[leader["k"] == k][cols].head(8).to_string(index=False))
    print("\nPairwise:")
    print(pair_df.to_string(index=False))


if __name__ == "__main__":
    main()
