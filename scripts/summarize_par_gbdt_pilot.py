#!/usr/bin/env python
"""Compare PAR residual tree pilot against existing cold-start baselines."""
import argparse
from pathlib import Path

import pandas as pd


def add_rank_table(df: pd.DataFrame, out_dir: Path, name: str):
    combo = df.copy()
    combo["rank"] = combo.groupby(["target", "k"])["MAE"].rank(method="min")
    rows = combo.groupby(["method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        worst_MAE=("MAE", "max"),
        mean_rank=("rank", "mean"),
        top1_rate=("rank", lambda x: float((x == 1).mean())),
        top3_rate=("rank", lambda x: float((x <= 3).mean())),
    ).reset_index().sort_values(["k", "mean_rank", "mean_MAE"])
    rows.to_csv(out_dir / f"table_{name}_ranking.csv", index=False)
    print(f"\n{name} ranking")
    print(rows.to_string(index=False))
    return rows


def pairwise(combo: pd.DataFrame, out_dir: Path, candidate_methods: list[str], baselines: list[str]):
    piv = combo.pivot_table(index=["target", "k"], columns="method", values="MAE", aggfunc="mean")
    rows = []
    for cand in candidate_methods:
        if cand not in piv.columns:
            continue
        for base in baselines:
            if base not in piv.columns:
                continue
            delta = piv[cand] - piv[base]
            tmp = delta.reset_index(name="delta").dropna()
            for k, g in tmp.groupby("k"):
                rows.append({
                    "candidate": cand,
                    "baseline": base,
                    "k": int(k),
                    "mean_delta": float(g["delta"].mean()),
                    "median_delta": float(g["delta"].median()),
                    "worst_delta": float(g["delta"].max()),
                    "best_delta": float(g["delta"].min()),
                    "win_rate": float((g["delta"] < 0).mean()),
                    "negative_transfer_rate": float((g["delta"] > 0).mean()),
                    "n": int(len(g)),
                })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_par_pairwise_deltas.csv", index=False)
    print("\nPairwise deltas: candidate minus baseline")
    if not out.empty:
        print(out.to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par-dir", default="results/par_gbdt_pilot_bdg2/aggregate")
    ap.add_argument("--paper-table", default="results/phase2_paper/expanded_fast/forecasting_paper/table_all_forecasting_methods_building_level.csv")
    ap.add_argument("--pa-table", default="results/pa_srft_pilot_bdg2/aggregate/table_pa_srft_pilot.csv")
    ap.add_argument("--output-dir", default="results/par_gbdt_pilot_bdg2/aggregate")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    par = pd.read_csv(Path(args.par_dir) / "table_par_gbdt_pilot_building_level.csv")
    par = par[["target", "method", "k", "MAE", "RMSE", "sMAPE", "sigma_err"]].copy()

    paper = pd.read_csv(args.paper_table)
    keep = [
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.3",
    ]
    base = paper[paper["method"].isin(keep)][["target", "method", "k", "MAE", "RMSE", "sMAPE", "sigma_err"]].copy()

    frames = [par, base]
    pa_path = Path(args.pa_table)
    if pa_path.exists():
        pa = pd.read_csv(pa_path)
        pa = pa.groupby(["target", "method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
        pa = pa[pa["method"].isin(["PA_SRFT", "Residual_SRFT", "Residual_M3"])]
        frames.append(pa)

    targets = set(par["target"].astype(str))
    ks = set(par["k"].astype(int))
    combo = pd.concat(frames, ignore_index=True)
    combo = combo[combo["target"].astype(str).isin(targets) & combo["k"].astype(int).isin(ks)]
    combo.to_csv(out_dir / "table_par_combined_building_level.csv", index=False)

    add_rank_table(combo, out_dir, "par_combined")
    pairwise(
        combo,
        out_dir,
        candidate_methods=sorted(par["method"].unique()),
        baselines=[
            "persistence",
            "random_forest_source_target",
            "extra_trees_source_target",
            "hist_gbdt_source_target",
            "dlinear_M3R_lambda0.3",
            "PA_SRFT",
        ],
    )


if __name__ == "__main__":
    main()
