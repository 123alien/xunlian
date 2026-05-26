#!/usr/bin/env python
"""Diagnose DLinear-SRFT failure modes in the expanded forecasting results."""
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("results/phase2_paper/expanded_fast")
OUT = BASE / "diagnostics"


def safe_ratio(a, b):
    return a / np.where(np.abs(b) < 1e-8, np.nan, b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(BASE / "forecasting_paper/table_all_forecasting_methods_building_level.csv")
    feat = pd.read_csv(BASE / "similarity/table_building_similarity_features.csv")
    feat = feat.rename(columns={"building_id": "target"})
    b = b.merge(feat, on="target", how="left")
    b["nMAE_pct"] = b["MAE"] / b["mean_energy"] * 100
    b["sigma_rel_pct"] = b["sigma_err"] / b["mean_energy"] * 100

    focus_methods = [
        "persistence",
        "seasonal_naive_24",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.3",
        "lstm_M3_pretrain_ft",
        "patchtst_M3_pretrain_ft",
    ]
    focus = b[b["method"].isin(focus_methods)].copy()

    # Wide table for pairwise gaps.
    wide = focus.pivot_table(
        index=["target", "k", "building_type", "mean_energy", "std_energy", "cv_energy",
               "daily_profile_range", "weekly_profile_range"],
        columns="method",
        values=["MAE", "nMAE_pct", "rank_MAE", "sMAPE", "sigma_err"],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["srft_minus_m3_MAE"] = wide["MAE_dlinear_M3R_lambda0.3"] - wide["MAE_dlinear_M3_pretrain_ft"]
    wide["srft_minus_rf_MAE"] = wide["MAE_dlinear_M3R_lambda0.3"] - wide["MAE_random_forest_source_target"]
    wide["srft_minus_persistence_MAE"] = wide["MAE_dlinear_M3R_lambda0.3"] - wide["MAE_persistence"]
    wide["srft_over_persistence_ratio"] = safe_ratio(
        wide["MAE_dlinear_M3R_lambda0.3"].to_numpy(),
        wide["MAE_persistence"].to_numpy(),
    )
    wide["rf_over_persistence_ratio"] = safe_ratio(
        wide["MAE_random_forest_source_target"].to_numpy(),
        wide["MAE_persistence"].to_numpy(),
    )
    wide["srft_failure_vs_rf"] = wide["srft_minus_rf_MAE"] > 5
    wide["srft_failure_vs_persistence"] = wide["srft_minus_persistence_MAE"] > 5
    wide.to_csv(OUT / "table_dlinear_pairwise_gaps.csv", index=False)

    failure = wide.sort_values("srft_minus_rf_MAE", ascending=False)
    failure.to_csv(OUT / "table_dlinear_failure_cases_ranked.csv", index=False)

    # Building-level worst-case summary across k.
    building_summary = wide.groupby(["target", "building_type"]).agg(
        mean_energy=("mean_energy", "first"),
        cv_energy=("cv_energy", "first"),
        daily_profile_range=("daily_profile_range", "first"),
        weekly_profile_range=("weekly_profile_range", "first"),
        mean_srft_mae=("MAE_dlinear_M3R_lambda0.3", "mean"),
        median_srft_mae=("MAE_dlinear_M3R_lambda0.3", "median"),
        mean_rf_mae=("MAE_random_forest_source_target", "mean"),
        mean_persistence_mae=("MAE_persistence", "mean"),
        max_srft_minus_rf=("srft_minus_rf_MAE", "max"),
        max_srft_minus_persistence=("srft_minus_persistence_MAE", "max"),
        mean_srft_nmae=("nMAE_pct_dlinear_M3R_lambda0.3", "mean"),
        mean_persistence_nmae=("nMAE_pct_persistence", "mean"),
        failure_count_vs_rf=("srft_failure_vs_rf", "sum"),
        failure_count_vs_persistence=("srft_failure_vs_persistence", "sum"),
    ).reset_index().sort_values("max_srft_minus_rf", ascending=False)
    building_summary.to_csv(OUT / "table_dlinear_failure_by_building.csv", index=False)

    # Type-level reporting.
    type_summary = focus.groupby(["building_type", "k", "method"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_nMAE_pct=("nMAE_pct", "mean"),
        median_nMAE_pct=("nMAE_pct", "median"),
        mean_rank=("rank_MAE", "mean"),
        n_buildings=("target", "nunique"),
    ).reset_index()
    type_summary.to_csv(OUT / "table_performance_by_building_type.csv", index=False)

    # Persistence-dominant / near-zero-like cases, inferred from very low persistence error.
    p = b[b["method"] == "persistence"].copy()
    p["persistence_nmae_pct"] = p["nMAE_pct"]
    pdom = p[p["rank_MAE"] <= 1].sort_values(["k", "persistence_nmae_pct"])
    pdom.to_csv(OUT / "table_persistence_top1_cases.csv", index=False)
    psmall = p[p["persistence_nmae_pct"] < 1.0].sort_values(["k", "persistence_nmae_pct"])
    psmall.to_csv(OUT / "table_low_persistence_error_cases.csv", index=False)

    # Correlation of SRFT gaps with building features.
    corr_cols = [
        "mean_energy", "std_energy", "cv_energy", "daily_profile_range",
        "weekly_profile_range", "MAE_persistence", "nMAE_pct_persistence",
        "srft_minus_rf_MAE", "srft_minus_persistence_MAE",
        "nMAE_pct_dlinear_M3R_lambda0.3",
    ]
    corr_rows = []
    for k, grp in wide.groupby("k"):
        c = grp[corr_cols].corr(numeric_only=True)
        for target_metric in ["srft_minus_rf_MAE", "srft_minus_persistence_MAE", "nMAE_pct_dlinear_M3R_lambda0.3"]:
            for feature in ["mean_energy", "std_energy", "cv_energy", "daily_profile_range", "weekly_profile_range", "MAE_persistence", "nMAE_pct_persistence"]:
                corr_rows.append({
                    "k": k,
                    "target_metric": target_metric,
                    "feature": feature,
                    "pearson_r": c.loc[target_metric, feature],
                })
    pd.DataFrame(corr_rows).to_csv(OUT / "table_failure_feature_correlations.csv", index=False)

    # Compact text report.
    report = []
    report.append("# DLinear Failure Diagnostics\n")
    report.append("## Worst SRFT-vs-RF cases\n")
    cols = [
        "target", "k", "building_type", "mean_energy", "cv_energy",
        "MAE_dlinear_M3R_lambda0.3", "MAE_random_forest_source_target",
        "MAE_persistence", "srft_minus_rf_MAE", "srft_minus_persistence_MAE",
        "nMAE_pct_dlinear_M3R_lambda0.3", "nMAE_pct_persistence",
    ]
    report.append(failure[cols].head(20).to_string(index=False))
    report.append("\n\n## Worst buildings averaged across k\n")
    report.append(building_summary.head(12).to_string(index=False))
    report.append("\n\n## Persistence-dominant counts by k\n")
    counts = pdom.groupby("k")["target"].nunique().reset_index(name="n_persistence_top1")
    report.append(counts.to_string(index=False))
    (OUT / "dlinear_failure_diagnostics.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[diagnose] saved diagnostics to {OUT}")
    print(building_summary.head(10).to_string(index=False))
    print("\nWorst cases:")
    print(failure[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
