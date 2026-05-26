#!/usr/bin/env python
"""Generate forecasting-focused paper tables and figures."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_md_table(df: pd.DataFrame, path: Path):
    if df.empty:
        path.write_text("", encoding="utf-8")
        return
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def mean_se(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return ""
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return f"{vals.mean():.3f} +/- {se:.3f}"


def build_summary_tables(agg: Path, stats: Path, out: Path):
    few = pd.read_csv(agg / "table_few_shot_transfer_all_models.csv")
    building_claim = pd.read_csv(stats / "table_building_level_claim_summary.csv")
    building_delta = pd.read_csv(stats / "table_building_level_m2_m3_deltas.csv")

    main_rows = []
    methods = ["M2_target_only", "M3_pretrain_ft"]
    for (model, k, method), g in (
        few[few["method"].isin(methods)]
        .groupby(["model", "k", "method"])
    ):
        if int(k) == 0:
            continue
        main_rows.append({
            "model": model,
            "k": int(k),
            "method": method,
            "n_runs": len(g),
            "MAE": mean_se(g["MAE"]),
            "RMSE": mean_se(g["RMSE"]),
            "sMAPE": mean_se(g["sMAPE"]),
            "sigma_err": mean_se(g["sigma_err"]),
        })
    main = pd.DataFrame(main_rows).sort_values(["model", "k", "method"])
    main.to_csv(out / "table_forecasting_main_summary.csv", index=False)
    save_md_table(main, out / "table_forecasting_main_summary.md")

    cols = [
        "model", "k", "n_buildings",
        "MAE_mean_delta", "MAE_ci95_low", "MAE_ci95_high", "MAE_wilcoxon_p", "MAE_win_rate",
        "sigma_err_mean_delta", "sigma_err_ci95_low", "sigma_err_ci95_high",
        "sigma_err_wilcoxon_p", "sigma_err_win_rate",
    ]
    claim = building_claim[cols].copy()
    claim.to_csv(out / "table_forecasting_building_level_claims.csv", index=False)
    save_md_table(claim.round(4), out / "table_forecasting_building_level_claims.md")

    neg = building_delta.sort_values("delta_MAE", ascending=False).head(20)
    neg.to_csv(out / "table_forecasting_weakest_buildings.csv", index=False)
    save_md_table(neg.round(4), out / "table_forecasting_weakest_buildings.md")
    return few, building_claim, building_delta


def plot_mae_sigma(few: pd.DataFrame, out: Path):
    plot_df = few[
        few["method"].isin(["M2_target_only", "M3_pretrain_ft"])
        & few["k"].isin([3, 7, 14])
    ].copy()
    labels = {"M2_target_only": "Target-only", "M3_pretrain_ft": "Pretrain+FT"}
    colors = {"M2_target_only": "#5f6368", "M3_pretrain_ft": "#1f77b4"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row, model in enumerate(["dlinear", "lstm"]):
        for col, metric in enumerate(["MAE", "sigma_err"]):
            ax = axes[row, col]
            sub = plot_df[plot_df["model"] == model]
            for method in ["M2_target_only", "M3_pretrain_ft"]:
                g = sub[sub["method"] == method].groupby("k")[metric]
                mean = g.mean()
                se = g.std(ddof=1) / np.sqrt(g.count())
                ax.errorbar(mean.index, mean.values, yerr=se.values,
                            marker="o", linewidth=2, capsize=3,
                            color=colors[method], label=labels[method])
            ax.set_title(f"{model.upper()} - {metric}")
            ax.set_ylabel(metric)
            ax.set_xticks([3, 7, 14])
            ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Target data budget (days)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure_forecasting_mae_sigma_err.png", dpi=300)
    plt.close(fig)


def plot_building_deltas(building_delta: pd.DataFrame, out: Path):
    for model in ["dlinear", "lstm"]:
        sub = building_delta[(building_delta["model"] == model) & (building_delta["k"] == 3)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("delta_MAE")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(sub["target"], sub["delta_MAE"], color="#2f6f9f")
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_xlabel("M3 - M2 MAE delta")
        ax.set_title(f"Building-level MAE transfer gains at k=3 ({model.upper()})")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"figure_building_delta_mae_{model}_k3.png", dpi=300)
        plt.close(fig)


def write_brief(building_claim: pd.DataFrame, out: Path):
    lines = [
        "# Forecasting Paper Results Brief",
        "",
        "## Main Claim",
        "",
        "Cross-building pretraining followed by few-shot target adaptation consistently improves cold-start building electricity forecasting and residual stability.",
        "",
        "## Building-Level Evidence",
        "",
    ]
    for _, r in building_claim.iterrows():
        lines.append(
            f"- {r['model']} k={int(r['k'])}: "
            f"MAE delta={r['MAE_mean_delta']:.3f} "
            f"[{r['MAE_ci95_low']:.3f}, {r['MAE_ci95_high']:.3f}], "
            f"MAE win rate={r['MAE_win_rate']:.1%}, "
            f"Wilcoxon p={r['MAE_wilcoxon_p']:.4f}; "
            f"sigma_err delta={r['sigma_err_mean_delta']:.3f}."
        )
    lines += [
        "",
        "## Claim Boundary",
        "",
        "Anomaly detection should be framed as a downstream motivation or exploratory extension. The current detection metrics do not support a universal anomaly-detection improvement claim.",
    ]
    (out / "forecasting_results_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--stats-dir", default="results/phase2_paper/forecasting_paper")
    ap.add_argument("--output-dir", default="results/phase2_paper/forecasting_assets")
    args = ap.parse_args()

    agg = Path(args.aggregate_dir)
    stats = Path(args.stats_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    few, building_claim, building_delta = build_summary_tables(agg, stats, out)
    plot_mae_sigma(few, out)
    plot_building_deltas(building_delta, out)
    write_brief(building_claim, out)
    print(f"[forecasting-assets] saved to {out}")


if __name__ == "__main__":
    main()
