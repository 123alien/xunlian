#!/usr/bin/env python
import pandas as pd


def cofactor():
    p = "results/cofactor_external/pilot/forecasting_paper/table_forecasting_leaderboard.csv"
    leader = pd.read_csv(p)
    cols = ["method", "mean_MAE", "median_MAE", "mean_rank_MAE", "top1_rate", "top3_rate"]
    print("COFACTOR leaderboard")
    for k in [3, 7, 14, 30]:
        print(f"\nK={k}")
        print(leader[leader["k"] == k][cols].head(10).to_string(index=False))


def pa_srft():
    pa = pd.read_csv("results/pa_srft_pilot_bdg2/aggregate/table_pa_srft_pilot.csv")
    print("\nPA table", pa.shape)
    print(pa.groupby(["method", "k"])[["MAE", "RMSE", "sigma_err"]].mean().reset_index().to_string(index=False))

    base = pd.read_csv("results/phase2_paper/expanded_fast/forecasting_paper/table_all_forecasting_methods_building_level.csv")
    pa_b = pa.groupby(["target", "k", "method"])["MAE"].mean().reset_index()
    existing_methods = [
        "persistence",
        "random_forest_source_target",
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.3",
    ]
    existing = base[base["method"].isin(existing_methods)][["target", "k", "method", "MAE"]]
    combo = pd.concat([existing, pa_b], ignore_index=True)
    combo = combo[combo["target"].isin(pa_b["target"].unique()) & combo["k"].isin(pa_b["k"].unique())]
    combo["rank"] = combo.groupby(["target", "k"])["MAE"].rank(method="min")
    out = combo.groupby(["method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_rank=("rank", "mean"),
        top1=("rank", lambda x: float((x == 1).mean())),
        top3=("rank", lambda x: float((x <= 3).mean())),
    ).reset_index().sort_values(["k", "mean_rank", "mean_MAE"])
    print("\nBDG2 PA pilot combined ranking")
    print(out.to_string(index=False))

    # Worst-case reduction against direct DLinear and original SRFT.
    piv = combo.pivot_table(index=["target", "k"], columns="method", values="MAE")
    for method in ["Residual_M3", "Residual_SRFT", "PA_SRFT"]:
        if method in piv.columns and "dlinear_M3R_lambda0.3" in piv.columns:
            delta = piv[method] - piv["dlinear_M3R_lambda0.3"]
            print(f"\n{method} minus original SRFT")
            print(delta.reset_index(name="delta").groupby("k")["delta"].agg(["mean", "median", "max", "min", lambda x: float((x < 0).mean())]).to_string())
        if method in piv.columns and "dlinear_M3_pretrain_ft" in piv.columns:
            delta = piv[method] - piv["dlinear_M3_pretrain_ft"]
            print(f"\n{method} minus DLinear M3")
            print(delta.reset_index(name="delta").groupby("k")["delta"].agg(["mean", "median", "max", "min", lambda x: float((x < 0).mean())]).to_string())


def main():
    cofactor()
    pa_srft()


if __name__ == "__main__":
    main()
