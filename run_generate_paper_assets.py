#!/usr/bin/env python
"""Generate paper-ready tables and figures from Phase 2 results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_se(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return ""
    mean = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return f"{mean:.3f} +/- {se:.3f}"


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


def build_forecasting_tables(agg: Path, out: Path):
    few = pd.read_csv(agg / "table_few_shot_transfer_all_models.csv")
    pairs = pd.read_csv(agg / "table_m2_m3_pairwise_summary_all_models.csv")

    methods = ["M2_target_only", "M3_pretrain_ft"]
    rows = []
    for (model, k, method), g in few[few["method"].isin(methods)].groupby(["model", "k", "method"]):
        rows.append({
            "model": model,
            "k": int(k),
            "method": method,
            "n": len(g),
            "MAE": mean_se(g["MAE"]),
            "RMSE": mean_se(g["RMSE"]),
            "sigma_err": mean_se(g["sigma_err"]),
            "sigma_score": mean_se(g["sigma_score"]),
        })
    summary = pd.DataFrame(rows).sort_values(["model", "k", "method"])
    summary.to_csv(out / "table_main_forecasting_summary.csv", index=False)
    save_md_table(summary, out / "table_main_forecasting_summary.md")

    claim = pairs.groupby(["model", "k"]).agg(
        n=("target", "count"),
        mae_win_rate=("M3_wins_MAE", "mean"),
        sigma_err_win_rate=("M3_wins_sigma_err", "mean"),
        sigma_score_win_rate=("M3_wins_sigma_score", "mean"),
        mean_delta_MAE=("delta_MAE", "mean"),
        mean_delta_sigma_err=("delta_sigma_err", "mean"),
        mean_delta_sigma_score=("delta_sigma_score", "mean"),
    ).reset_index()
    claim.to_csv(out / "table_main_claim_summary.csv", index=False)
    save_md_table(claim.round(4), out / "table_main_claim_summary.md")

    neg = pairs.sort_values("delta_MAE", ascending=False).head(20)
    neg.to_csv(out / "table_negative_transfer_top20.csv", index=False)
    save_md_table(neg.round(4), out / "table_negative_transfer_top20.md")

    return few, pairs, claim


def build_scoring_tables(agg: Path, out: Path):
    scoring = pd.read_csv(agg / "table_scoring_ablation.csv")
    claim = pd.read_csv(agg / "table_scoring_ablation_claim_summary.csv")

    claim_sorted = claim.sort_values(
        ["model", "k", "mean_delta_F1", "mean_delta_AUPRC"],
        ascending=[True, True, False, False],
    )
    best_f1 = claim_sorted.groupby(["model", "k"]).head(3)
    best_f1.to_csv(out / "table_scoring_top3_by_f1_delta.csv", index=False)
    save_md_table(best_f1.round(4), out / "table_scoring_top3_by_f1_delta.md")

    abs_summary = scoring.groupby(["model", "k", "method", "scorer"])[
        ["Precision", "Recall", "F1", "AUROC", "AUPRC", "FAR"]
    ].mean().reset_index()
    abs_summary.to_csv(out / "table_scoring_absolute_summary.csv", index=False)

    selected = abs_summary[
        abs_summary["scorer"].isin(["rolling_mad_q975", "ewma_q975", "cusum_q975", "hybrid_q975"])
    ].copy()
    selected.to_csv(out / "table_scoring_selected_summary.csv", index=False)
    save_md_table(selected.round(4), out / "table_scoring_selected_summary.md")

    type_cols = [c for c in scoring.columns if c.startswith("Recall_")]
    type_rows = []
    if type_cols:
        filt = scoring[
            (scoring["method"] == "M3_pretrain_ft")
            & (scoring["k"] == 7)
            & (scoring["scorer"].isin(["rolling_mad_q975", "ewma_q975", "cusum_q975", "hybrid_q975"]))
        ]
        type_summary = filt.groupby(["model", "scorer"])[type_cols].mean().reset_index()
        type_summary.to_csv(out / "table_per_type_recall_m3_k7.csv", index=False)
        save_md_table(type_summary.round(4), out / "table_per_type_recall_m3_k7.md")
        type_rows = type_summary

    return scoring, claim, selected, type_rows


def plot_fewshot_curves(few: pd.DataFrame, out: Path):
    plot_df = few[few["method"].isin(["M2_target_only", "M3_pretrain_ft"])].copy()
    method_labels = {"M2_target_only": "Target-only", "M3_pretrain_ft": "Pretrain+FT"}
    colors = {"M2_target_only": "#555555", "M3_pretrain_ft": "#1f77b4"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    metrics = [("MAE", "MAE"), ("sigma_err", "Residual std")]
    for row, model in enumerate(["lstm", "dlinear"]):
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            sub = plot_df[plot_df["model"] == model]
            for method in ["M2_target_only", "M3_pretrain_ft"]:
                g = sub[sub["method"] == method].groupby("k")[metric]
                mean = g.mean()
                se = g.std(ddof=1) / np.sqrt(g.count())
                ax.errorbar(mean.index, mean.values, yerr=se.values,
                            marker="o", linewidth=2, capsize=3,
                            color=colors[method], label=method_labels[method])
            ax.set_title(f"{model.upper()} - {ylabel}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.set_xticks([3, 7, 14])
    for ax in axes[-1, :]:
        ax.set_xlabel("Few-shot target data (days)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure_fewshot_mae_sigma_err.png", dpi=300)
    plt.close(fig)


def plot_scoring_ablation(scoring: pd.DataFrame, out: Path):
    selected = scoring[
        scoring["scorer"].isin(["rolling_mad_q975", "ewma_q975", "cusum_q975", "hybrid_q975"])
        & scoring["method"].isin(["M2_target_only", "M3_pretrain_ft"])
    ].copy()
    selected["label"] = selected["method"].map({
        "M2_target_only": "M2",
        "M3_pretrain_ft": "M3",
    }) + " / " + selected["scorer"].str.replace("_q975", "", regex=False)

    for model in ["lstm", "dlinear"]:
        sub = selected[(selected["model"] == model) & (selected["k"] == 7)]
        if sub.empty:
            continue
        summary = sub.groupby(["label"])[["F1", "AUPRC", "FAR"]].mean()
        order = summary.sort_values("F1", ascending=False).index
        summary = summary.loc[order]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, ["F1", "AUPRC", "FAR"]):
            ax.barh(summary.index, summary[metric], color="#4c78a8")
            ax.set_title(metric)
            ax.grid(axis="x", alpha=0.25)
            if metric != "F1":
                ax.set_yticklabels([])
        fig.suptitle(f"Scoring ablation at k=7 ({model.upper()})")
        fig.tight_layout()
        fig.savefig(out / f"figure_scoring_ablation_{model}_k7.png", dpi=300)
        plt.close(fig)


def plot_per_type_recall(type_summary, out: Path):
    if isinstance(type_summary, list) or type_summary is None or len(type_summary) == 0:
        return
    type_cols = [c for c in type_summary.columns if c.startswith("Recall_")]
    labels = [c.replace("Recall_", "") for c in type_cols]
    for model in ["lstm", "dlinear"]:
        sub = type_summary[type_summary["model"] == model]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(labels))
        width = 0.18
        scorers = list(sub["scorer"])
        for i, scorer in enumerate(scorers):
            vals = sub[sub["scorer"] == scorer][type_cols].iloc[0].values
            ax.bar(x + (i - len(scorers) / 2) * width + width / 2, vals,
                   width=width, label=scorer.replace("_q975", ""))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-type recall for M3 at k=7 ({model.upper()})")
        ax.legend(frameon=False, ncols=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"figure_per_type_recall_m3_{model}_k7.png", dpi=300)
        plt.close(fig)


def write_narrative(claim: pd.DataFrame, scoring_claim: pd.DataFrame, out: Path):
    lines = [
        "# Paper Results Brief",
        "",
        "## Main Finding",
        "",
        "Pretrain+fine-tune (M3) consistently improves few-shot forecasting accuracy and residual stability compared with target-only training (M2). This is the strongest supported claim.",
        "",
        "## Forecasting And Stability",
        "",
    ]
    for _, r in claim.iterrows():
        lines.append(
            f"- {r['model']} k={int(r['k'])}: MAE win rate={r['mae_win_rate']:.1%}, "
            f"sigma_err win rate={r['sigma_err_win_rate']:.1%}, "
            f"mean delta MAE={r['mean_delta_MAE']:.3f}, "
            f"mean delta sigma_err={r['mean_delta_sigma_err']:.3f}."
        )
    lines += [
        "",
        "## Anomaly Detection",
        "",
        "The anomaly-detection story is more nuanced. Better residuals do not automatically produce better detection under every scoring strategy. EWMA and CUSUM improve F1 in some settings, especially DLinear at smaller k, but AUPRC and FAR remain inconsistent.",
        "",
        "## Recommended Claim",
        "",
        "Use: Transfer learning improves cold-start forecasting and residual stability, providing a stronger residual foundation for anomaly detection. However, the conversion from residual stability to detection performance depends on scoring and threshold calibration.",
        "",
        "Avoid: Transfer learning universally improves anomaly detection.",
        "",
    ]
    out.joinpath("paper_results_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--output-dir", default="results/phase2_paper/paper_assets")
    args = ap.parse_args()

    agg = Path(args.aggregate_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    few, pairs, claim = build_forecasting_tables(agg, out)
    scoring, scoring_claim, selected, type_summary = build_scoring_tables(agg, out)
    plot_fewshot_curves(few, out)
    plot_scoring_ablation(scoring, out)
    plot_per_type_recall(type_summary, out)
    write_narrative(claim, scoring_claim, out)

    print(f"[paper-assets] saved to {out}")


if __name__ == "__main__":
    main()
