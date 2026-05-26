#!/usr/bin/env python
"""Summarize PA-MSR-only BDG2-120 lightweight validation.

This script compares the new PA-MSR-only run with the existing BDG2-120
classical baselines and earlier residual boosting baseline.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    "M9_REVIN_MSR": "PA-MSR",
    "PAR_HIST_GBDT_SW_CAL": "Prior residual boosting",
    "persistence": "Persistence",
    "random_forest_source_target": "RF-ST",
    "extra_trees_source_target": "ExtraTrees-ST",
    "hist_gbdt_source_target": "HistGBDT-ST",
    "seasonal_naive_24": "Seasonal naive",
}


def write_md(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for _, row in df.iterrows():
            vals = []
            for val in row:
                if isinstance(val, float):
                    vals.append(f"{val:.4g}")
                else:
                    vals.append(str(val))
            f.write("| " + " | ".join(vals) + " |\n")


def safe_wilcoxon_greater(x: pd.Series) -> float:
    try:
        from scipy.stats import wilcoxon

        vals = x.dropna().to_numpy(dtype=float)
        vals = vals[np.abs(vals) > 1e-12]
        if len(vals) == 0:
            return float("nan")
        # improvement = baseline - candidate; greater tests candidate is better.
        return float(wilcoxon(vals, alternative="greater").pvalue)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", default="results/pa_msr_bdg2_120_light/aggregate/table_pa_module_extensions.csv")
    ap.add_argument("--parboost", default="results/bdg2_robustness_120/parboost/aggregate/table_par_gbdt_pilot_building_level.csv")
    ap.add_argument("--baselines", default="results/bdg2_robustness_120/forecasting_baselines/table_forecasting_baselines_building_level.csv")
    ap.add_argument("--output-dir", default="results/pa_msr_bdg2_120_light/summary")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pa = pd.read_csv(args.pa)
    pa = pa[pa["method"] == "M9_REVIN_MSR"].copy()
    par = pd.read_csv(args.parboost)
    base = pd.read_csv(args.baselines)
    keep = [
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
        "seasonal_naive_24",
    ]
    base = base[base["method"].isin(keep)].copy()

    cols = ["target", "method", "k", "MAE", "RMSE", "sMAPE", "sigma_err"]
    combo = pd.concat([pa[cols], par[cols], base[cols]], ignore_index=True)
    combo["method_label"] = combo["method"].map(LABELS).fillna(combo["method"])
    combo["rank"] = combo.groupby(["target", "k"])["MAE"].rank(method="min")
    combo.to_csv(out / "table_pa_msr_bdg2_120_building_level.csv", index=False)

    leaderboard = combo.groupby(["method", "method_label", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        p90_MAE=("MAE", lambda s: float(s.quantile(0.90))),
        p95_MAE=("MAE", lambda s: float(s.quantile(0.95))),
        worst_MAE=("MAE", "max"),
        mean_rank=("rank", "mean"),
        top3_rate=("rank", lambda s: float((s <= 3).mean())),
        n=("MAE", "size"),
    ).reset_index().sort_values(["k", "mean_rank", "mean_MAE"])
    leaderboard.to_csv(out / "table_pa_msr_bdg2_120_leaderboard.csv", index=False)
    write_md(leaderboard, out / "table_pa_msr_bdg2_120_leaderboard.md")

    piv = combo.pivot_table(index=["target", "k"], columns="method", values="MAE", aggfunc="mean")
    rows = []
    for baseline in ["PAR_HIST_GBDT_SW_CAL"] + keep:
        if baseline not in piv.columns:
            continue
        tmp = (piv[baseline] - piv["M9_REVIN_MSR"]).reset_index(name="improvement")
        for k, g in tmp.dropna().groupby("k"):
            rows.append({
                "candidate": "PA-MSR",
                "baseline": LABELS.get(baseline, baseline),
                "k": int(k),
                "n": int(len(g)),
                "candidate_mean_MAE": float(piv["M9_REVIN_MSR"].reset_index().query("k == @k")["M9_REVIN_MSR"].mean()),
                "baseline_mean_MAE": float(piv[baseline].reset_index().query("k == @k")[baseline].mean()),
                "mean_improvement_MAE": float(g["improvement"].mean()),
                "median_improvement_MAE": float(g["improvement"].median()),
                "p10_improvement_MAE": float(g["improvement"].quantile(0.10)),
                "win_rate": float((g["improvement"] > 0).mean()),
                "negative_transfer_rate": float((g["improvement"] < 0).mean()),
                "wilcoxon_p_greater": safe_wilcoxon_greater(g["improvement"]),
            })
    pairwise = pd.DataFrame(rows).sort_values(["baseline", "k"])
    pairwise.to_csv(out / "table_pa_msr_bdg2_120_pairwise.csv", index=False)
    write_md(pairwise, out / "table_pa_msr_bdg2_120_pairwise.md")

    compact = leaderboard[leaderboard["method"].isin([
        "M9_REVIN_MSR",
        "PAR_HIST_GBDT_SW_CAL",
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
    ])].copy()
    print(compact.to_string(index=False))
    print(f"[summary] wrote {out}")


if __name__ == "__main__":
    main()
