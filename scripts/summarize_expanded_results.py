#!/usr/bin/env python
from pathlib import Path

import pandas as pd


BASE = Path("results/phase2_paper/expanded_fast")


def show_leaderboard():
    leader = pd.read_csv(BASE / "forecasting_paper/table_forecasting_leaderboard.csv")
    cols = ["method", "mean_MAE", "median_MAE", "mean_rank_MAE", "top1_rate", "top3_rate"]
    print("LEADERBOARD", leader.shape)
    for k in [3, 7, 14, 30]:
        print(f"\nK={k} TOP12")
        print(leader[leader["k"] == k][cols].head(12).to_string(index=False))


def show_transfer_pairs():
    allm = pd.read_csv(BASE / "forecasting_paper/table_all_forecasting_methods.csv")
    b = allm.groupby(["target", "k", "method"])[["MAE", "sigma_err"]].mean().reset_index()
    for model in ["lstm", "dlinear", "patchtst"]:
        m2 = f"{model}_M2_target_only"
        m3 = f"{model}_M3_pretrain_ft"
        piv = b[b["method"].isin([m2, m3])].pivot_table(index=["target", "k"], columns="method", values="MAE")
        if m2 not in piv or m3 not in piv:
            continue
        delta = piv[m3] - piv[m2]
        out = delta.reset_index(name="delta_MAE_M3_minus_M2").groupby("k")["delta_MAE_M3_minus_M2"].agg(
            mean_delta="mean",
            median_delta="median",
            win_rate=lambda x: float((x < 0).mean()),
            n="count",
        )
        print(f"\nTRANSFER {model} M3 vs M2")
        print(out.to_string())


def show_srft():
    allm = pd.read_csv(BASE / "forecasting_paper/table_all_forecasting_methods.csv")
    b = allm.groupby(["target", "k", "method"])[["MAE", "sigma_err"]].mean().reset_index()
    dmethods = [m for m in b["method"].unique() if m.startswith("dlinear_")]
    print("\nDLINEAR METHODS")
    dsum = b[b["method"].isin(dmethods)].groupby(["method", "k"])[["MAE", "sigma_err"]].mean().reset_index()
    print(dsum.sort_values(["k", "MAE"]).to_string(index=False))
    for lam in ["0.05", "0.1", "0.2", "0.3"]:
        m3 = "dlinear_M3_pretrain_ft"
        srft = f"dlinear_M3R_lambda{lam}"
        piv = b[b["method"].isin([m3, srft])].pivot_table(index=["target", "k"], columns="method", values="MAE")
        if m3 not in piv or srft not in piv:
            continue
        delta = piv[srft] - piv[m3]
        out = delta.reset_index(name="delta_MAE_SRFT_minus_M3").groupby("k")["delta_MAE_SRFT_minus_M3"].agg(
            mean_delta="mean",
            median_delta="median",
            win_rate=lambda x: float((x < 0).mean()),
            n="count",
        )
        print(f"\nSRFT lambda={lam} vs DLinear M3")
        print(out.to_string())


def show_similarity():
    sim = BASE / "similarity/table_dlinear_similarity_transfer_gains.csv"
    if not sim.exists():
        return
    df = pd.read_csv(sim)
    print("\nSIMILARITY")
    cols = [c for c in ["k", "gain_m3_vs_m2", "gain_srft_vs_m3", "nearest_distance", "mean_source_distance"] if c in df]
    print(df[cols].groupby("k").mean(numeric_only=True).to_string())
    for gain in ["gain_m3_vs_m2", "gain_srft_vs_m3"]:
        if gain in df:
            print(f"{gain} corr nearest_distance:", df[[gain, "nearest_distance"]].corr().iloc[0, 1])


def show_scale_checks():
    b = pd.read_csv(BASE / "forecasting_paper/table_all_forecasting_methods_building_level.csv")
    manifest = pd.read_csv(BASE / "building_manifest_24.csv")
    meta = manifest[["building_id", "building_type", "mean_energy", "cv_energy"]].rename(
        columns={"building_id": "target"}
    )
    b = b.merge(meta, on="target", how="left")
    b["nMAE_pct"] = b["MAE"] / b["mean_energy"] * 100
    methods = [
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.3",
        "lstm_M3_pretrain_ft",
        "patchtst_M3_pretrain_ft",
    ]
    print("\nBUILDING MANIFEST")
    print(manifest[["building_id", "building_type", "mean_energy", "cv_energy"]].to_string(index=False))
    for k in [3, 7, 14, 30]:
        print(f"\nK={k} NORMALIZED MAE %")
        s = (
            b[(b["k"] == k) & (b["method"].isin(methods))]
            .groupby("method")["nMAE_pct"]
            .agg(["mean", "median"])
            .sort_values("median")
        )
        print(s.to_string())
        print(f"\nK={k} TOP1 COUNTS")
        sub = b[b["k"] == k]
        best = sub.loc[sub.groupby("target")["MAE"].idxmin()]["method"].value_counts()
        print(best.to_string())


def main():
    show_leaderboard()
    show_transfer_pairs()
    show_srft()
    show_similarity()
    show_scale_checks()


if __name__ == "__main__":
    main()
