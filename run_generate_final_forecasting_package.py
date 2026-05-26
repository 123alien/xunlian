#!/usr/bin/env python
"""Generate final tables/figures for the forecasting-focused manuscript."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SELECTED_METHODS = [
    "persistence",
    "random_forest_source_target",
    "extra_trees_source_target",
    "dlinear_M2_target_only",
    "dlinear_M3_pretrain_ft",
    "dlinear_M3R_lambda0.2",
    "lstm_M2_target_only",
    "lstm_M3_pretrain_ft",
]

METHOD_LABELS = {
    "persistence": "Persistence",
    "random_forest_source_target": "RF source+target",
    "extra_trees_source_target": "ExtraTrees source+target",
    "dlinear_M2_target_only": "DLinear target-only",
    "dlinear_M3_pretrain_ft": "DLinear pretrain+FT",
    "dlinear_M3R_lambda0.2": "DLinear replay FT",
    "lstm_M2_target_only": "LSTM target-only",
    "lstm_M3_pretrain_ft": "LSTM pretrain+FT",
}


def save_md_table(df: pd.DataFrame, path: Path):
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def mean_se(vals: pd.Series) -> str:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return f"{vals.mean():.3f} +/- {se:.3f}"


def load_selected(compare_dir: Path) -> pd.DataFrame:
    combined = pd.read_csv(compare_dir / "table_all_forecasting_methods.csv")
    selected = combined[combined["method"].isin(SELECTED_METHODS)].copy()
    selected["method_label"] = selected["method"].map(METHOD_LABELS)
    return selected


def build_main_tables(selected: pd.DataFrame, stats_dir: Path, out: Path):
    rows = []
    for (method, label, k), g in selected.groupby(["method", "method_label", "k"]):
        rows.append({
            "method": label,
            "k": int(k),
            "n_runs": len(g),
            "MAE": mean_se(g["MAE"]),
            "RMSE": mean_se(g["RMSE"]),
            "sMAPE": mean_se(g["sMAPE"]),
            "sigma_err": mean_se(g["sigma_err"]),
        })
    main = pd.DataFrame(rows).sort_values(["k", "method"])
    main.to_csv(out / "table_final_method_summary.csv", index=False)
    save_md_table(main, out / "table_final_method_summary.md")

    bld = selected.groupby(["target", "k", "method", "method_label"])[
        ["MAE", "RMSE", "sMAPE", "sigma_err"]
    ].mean().reset_index()
    bld["rank_MAE"] = bld.groupby(["target", "k"])["MAE"].rank(method="min")
    leaderboard = bld.groupby(["method", "method_label", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_sigma_err=("sigma_err", "mean"),
        mean_rank_MAE=("rank_MAE", "mean"),
        top1_rate=("rank_MAE", lambda x: float(np.mean(x == 1))),
        top3_rate=("rank_MAE", lambda x: float(np.mean(x <= 3))),
    ).reset_index().sort_values(["k", "mean_rank_MAE", "mean_MAE"])
    leaderboard.to_csv(out / "table_final_leaderboard.csv", index=False)
    save_md_table(
        leaderboard[[
            "method_label", "k", "mean_MAE", "median_MAE",
            "mean_sigma_err", "mean_rank_MAE", "top1_rate", "top3_rate",
        ]].round(4),
        out / "table_final_leaderboard.md",
    )

    m3r_vs_m3 = pd.read_csv(stats_dir / "table_m3r_vs_m3_building_summary.csv")
    m3r_vs_m3 = m3r_vs_m3[(m3r_vs_m3["model"] == "dlinear") & (m3r_vs_m3["lambda"] == 0.2)]
    m3r_vs_m3.to_csv(out / "table_final_dlinear_m3r_vs_m3.csv", index=False)
    save_md_table(m3r_vs_m3.round(4), out / "table_final_dlinear_m3r_vs_m3.md")
    return bld, leaderboard


def plot_method_mae(selected: pd.DataFrame, out: Path):
    summary = selected.groupby(["method", "method_label", "k"])["MAE"].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    order = [
        "Persistence",
        "RF source+target",
        "ExtraTrees source+target",
        "DLinear target-only",
        "DLinear pretrain+FT",
        "DLinear replay FT",
        "LSTM target-only",
        "LSTM pretrain+FT",
    ]
    colors = {
        "Persistence": "#666666",
        "RF source+target": "#2ca02c",
        "ExtraTrees source+target": "#78b159",
        "DLinear target-only": "#9ecae1",
        "DLinear pretrain+FT": "#3182bd",
        "DLinear replay FT": "#08519c",
        "LSTM target-only": "#fdae6b",
        "LSTM pretrain+FT": "#e6550d",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, k in zip(axes, [3, 7, 14]):
        sub = summary[summary["k"] == k].set_index("method_label").reindex(order).dropna()
        y = np.arange(len(sub))
        ax.barh(y, sub["mean"], xerr=sub["se"], color=[colors[i] for i in sub.index], capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub.index if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(f"k={k} days")
        ax.set_xlabel("MAE")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figure_final_method_mae.png", dpi=300)
    plt.close(fig)


def plot_dlinear_transfer_curve(selected: pd.DataFrame, out: Path):
    methods = [
        "dlinear_M2_target_only",
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.2",
        "random_forest_source_target",
        "persistence",
    ]
    plot_df = selected[selected["method"].isin(methods)].copy()
    colors = {
        "DLinear target-only": "#9ecae1",
        "DLinear pretrain+FT": "#3182bd",
        "DLinear replay FT": "#08519c",
        "RF source+target": "#2ca02c",
        "Persistence": "#666666",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        label = METHOD_LABELS[method]
        g = plot_df[plot_df["method"] == method].groupby("k")["MAE"]
        mean = g.mean()
        se = g.std(ddof=1) / np.sqrt(g.count())
        ax.errorbar(mean.index, mean.values, yerr=se.values,
                    marker="o", linewidth=2, capsize=3,
                    label=label, color=colors[label])
    ax.set_xticks([3, 7, 14])
    ax.set_xlabel("Target data budget (days)")
    ax.set_ylabel("MAE")
    ax.set_title("Cold-start forecasting performance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure_final_dlinear_transfer_curve.png", dpi=300)
    plt.close(fig)


def plot_m3r_building_delta(stats_dir: Path, out: Path):
    m3r = pd.read_csv(stats_dir / "table_m3r_vs_m3_building_summary.csv")
    deltas = pd.read_csv(stats_dir / "table_all_forecasting_methods_building_level.csv")
    # Reconstruct per-building DLinear M3R-M3 delta from all-method table.
    wide = deltas[deltas["method"].isin(["dlinear_M3_pretrain_ft", "dlinear_M3R_lambda0.2"])].pivot_table(
        index=["target", "k"], columns="method", values="MAE"
    ).reset_index()
    wide["delta_MAE"] = wide["dlinear_M3R_lambda0.2"] - wide["dlinear_M3_pretrain_ft"]
    for k in [3, 7, 14]:
        sub = wide[wide["k"] == k].sort_values("delta_MAE")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(sub["target"], sub["delta_MAE"], color="#08519c")
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_xlabel("DLinear replay FT - pretrain+FT MAE")
        ax.set_title(f"Building-level effect of source replay (k={k})")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"figure_final_m3r_delta_k{k}.png", dpi=300)
        plt.close(fig)


def write_final_brief(leaderboard: pd.DataFrame, out: Path):
    lines = [
        "# Final Forecasting Package Brief",
        "",
        "## Main Line",
        "",
        "The manuscript is a cold-start building energy forecasting paper. The core claim is that cross-building transfer improves few-shot neural forecasting, and that source replay further strengthens DLinear transfer.",
        "",
        "## Key Takeaways",
        "",
        "- M3 pretrain+fine-tune strongly improves over target-only neural training.",
        "- DLinear replay fine-tuning with lambda=0.2 consistently improves over standard DLinear M3.",
        "- Source+target tree baselines are strong and must be reported honestly.",
        "- The strongest claim is not universal superiority, but robust neural transfer improvement plus competitive performance at k=7/k=14.",
        "",
        "## Best Methods By k",
        "",
    ]
    for k in [3, 7, 14]:
        sub = leaderboard[leaderboard["k"] == k].head(5)
        lines.append(f"### k={k}")
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['method_label']}: mean MAE={r['mean_MAE']:.3f}, "
                f"mean rank={r['mean_rank_MAE']:.2f}, top3={r['top3_rate']:.1%}."
            )
        lines.append("")
    (out / "final_forecasting_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-dir", default="results/phase2_paper/forecasting_paper")
    ap.add_argument("--output-dir", default="results/phase2_paper/final_forecasting_package")
    args = ap.parse_args()

    compare = Path(args.compare_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = load_selected(compare)
    bld, leaderboard = build_main_tables(selected, compare, out)
    plot_method_mae(selected, out)
    plot_dlinear_transfer_curve(selected, out)
    plot_m3r_building_delta(compare, out)
    write_final_brief(leaderboard, out)
    print(f"[final-forecasting] saved to {out}")


if __name__ == "__main__":
    main()
