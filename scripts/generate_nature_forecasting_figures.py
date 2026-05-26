#!/usr/bin/env python
"""Publication-grade forecasting figures using Python/matplotlib only."""
from pathlib import Path
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


OUT = Path("results/phase2_paper/nature_figures")
PAPER = Path("results/phase2_paper/forecasting_paper")
FINAL = Path("results/phase2_paper/final_forecasting_package")


METHOD_ORDER = [
    "persistence",
    "random_forest_source_target",
    "extra_trees_source_target",
    "dlinear_M2_target_only",
    "dlinear_M3_pretrain_ft",
    "dlinear_M3R_lambda0.2",
    "lstm_M2_target_only",
    "lstm_M3_pretrain_ft",
]

LABELS = {
    "persistence": "Persistence",
    "random_forest_source_target": "RF source+target",
    "extra_trees_source_target": "ExtraTrees source+target",
    "dlinear_M2_target_only": "DLinear target-only",
    "dlinear_M3_pretrain_ft": "DLinear pretrain+FT",
    "dlinear_M3R_lambda0.2": "DLinear-SRFT",
    "lstm_M2_target_only": "LSTM target-only",
    "lstm_M3_pretrain_ft": "LSTM pretrain+FT",
}

PALETTE = {
    "persistence": "#6f6f6f",
    "random_forest_source_target": "#3f7f5f",
    "extra_trees_source_target": "#8aae68",
    "dlinear_M2_target_only": "#9ecae1",
    "dlinear_M3_pretrain_ft": "#4292c6",
    "dlinear_M3R_lambda0.2": "#084594",
    "lstm_M2_target_only": "#fdd0a2",
    "lstm_M3_pretrain_ft": "#e6550d",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_pub(fig, stem: str, width_in=None, height_in=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if width_in and height_in:
        fig.set_size_inches(width_in, height_in)
    for ext in ["svg", "pdf", "png", "tiff"]:
        dpi = 600 if ext in {"png", "tiff"} else None
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)


def panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def load_data():
    all_methods = pd.read_csv(PAPER / "table_all_forecasting_methods.csv")
    selected = all_methods[all_methods["method"].isin(METHOD_ORDER)].copy()
    selected["method_label"] = selected["method"].map(LABELS)
    building = pd.read_csv(PAPER / "table_all_forecasting_methods_building_level.csv")
    building = building[building["method"].isin(METHOD_ORDER)].copy()
    building["method_label"] = building["method"].map(LABELS)
    m3r = pd.read_csv(PAPER / "table_m3r_vs_m3_building_summary.csv")
    return selected, building, m3r


def plot_figure1(selected: pd.DataFrame, building: pd.DataFrame):
    """Overall comparison across target-data budgets."""
    source = selected.groupby(["method", "method_label", "k"])["MAE"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    source["se"] = source["std"] / np.sqrt(source["n"])
    source.to_csv(OUT / "source_data_figure1a_mae_curves.csv", index=False)

    sigma = selected.groupby(["method", "method_label", "k"])["sigma_err"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    sigma["se"] = sigma["std"] / np.sqrt(sigma["n"])
    sigma.to_csv(OUT / "source_data_figure1b_sigma_curves.csv", index=False)

    rank = building.copy()
    rank["rank_MAE"] = rank.groupby(["target", "k"])["MAE"].rank(method="min")
    leader = rank.groupby(["method", "method_label", "k"]).agg(
        mean_rank=("rank_MAE", "mean"),
        top3_rate=("rank_MAE", lambda x: float(np.mean(x <= 3))),
    ).reset_index()
    leader.to_csv(OUT / "source_data_figure1c_rank.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], figure=fig)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    highlight = [
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "dlinear_M3R_lambda0.2",
        "dlinear_M3_pretrain_ft",
        "lstm_M3_pretrain_ft",
    ]
    for method in highlight:
        sub = source[source["method"] == method].sort_values("k")
        lw = 2.2 if method == "dlinear_M3R_lambda0.2" else 1.4
        z = 5 if method == "dlinear_M3R_lambda0.2" else 2
        ax_a.errorbar(sub["k"], sub["mean"], yerr=sub["se"],
                      marker="o", linewidth=lw, markersize=3.5, capsize=2.5,
                      color=PALETTE[method], label=LABELS[method], zorder=z)
    ax_a.set_xlabel("Target data budget (days)")
    ax_a.set_ylabel("MAE")
    ax_a.set_xticks([3, 7, 14])
    ax_a.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax_a.set_title("Forecasting error across cold-start budgets")
    ax_a.legend(loc="upper right", ncol=1)
    panel_label(ax_a, "a")

    for method in ["dlinear_M3R_lambda0.2", "random_forest_source_target",
                   "extra_trees_source_target", "persistence"]:
        sub = sigma[sigma["method"] == method].sort_values("k")
        ax_b.errorbar(sub["k"], sub["mean"], yerr=sub["se"],
                      marker="o", linewidth=1.5, markersize=3.2, capsize=2.3,
                      color=PALETTE[method], label=LABELS[method])
    ax_b.set_xlabel("Target data budget (days)")
    ax_b.set_ylabel("Residual std.")
    ax_b.set_xticks([3, 7, 14])
    ax_b.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax_b.set_title("Residual stability")
    panel_label(ax_b, "b")

    sub = leader[leader["method"].isin([
        "dlinear_M3R_lambda0.2",
        "random_forest_source_target",
        "extra_trees_source_target",
        "persistence",
    ])]
    pivot = sub.pivot(index="method_label", columns="k", values="top3_rate")
    pivot = pivot.reindex([LABELS[m] for m in [
        "dlinear_M3R_lambda0.2",
        "random_forest_source_target",
        "extra_trees_source_target",
        "persistence",
    ]])
    im = ax_c.imshow(pivot.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax_c.set_xticks(np.arange(len(pivot.columns)))
    ax_c.set_xticklabels([str(c) for c in pivot.columns])
    ax_c.set_yticks(np.arange(len(pivot.index)))
    ax_c.set_yticklabels(pivot.index)
    ax_c.set_xlabel("Target data budget (days)")
    ax_c.set_title("Top-3 rate across buildings")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax_c.text(j, i, f"{pivot.values[i, j]:.0%}",
                      ha="center", va="center", fontsize=6.5,
                      color="white" if pivot.values[i, j] > 0.62 else "#222222")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03)
    cbar.set_label("Top-3 rate", rotation=270, labelpad=10)
    panel_label(ax_c, "c")

    fig.tight_layout(pad=1.1)
    save_pub(fig, "figure1_method_comparison", 7.2, 4.6)
    plt.close(fig)


def plot_figure2(building: pd.DataFrame):
    """DLinear-SRFT effect versus standard DLinear transfer."""
    wide = building[building["method"].isin([
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.2",
        "dlinear_M2_target_only",
    ])].pivot_table(index=["target", "k"], columns="method", values="MAE").reset_index()
    wide["delta_srft_vs_m3"] = wide["dlinear_M3R_lambda0.2"] - wide["dlinear_M3_pretrain_ft"]
    wide["delta_m3_vs_m2"] = wide["dlinear_M3_pretrain_ft"] - wide["dlinear_M2_target_only"]
    wide.to_csv(OUT / "source_data_figure2_dlinear_deltas.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.1, 1], figure=fig)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # paired deltas per k with all buildings
    positions = {3: 0, 7: 1, 14: 2}
    rng = np.random.default_rng(42)
    for k, x in positions.items():
        vals = wide[wide["k"] == k]["delta_srft_vs_m3"].to_numpy()
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax_a.scatter(np.full(len(vals), x) + jitter, vals, s=18,
                     color="#084594", alpha=0.78, edgecolor="white", linewidth=0.3)
        mean = float(np.mean(vals))
        ax_a.plot([x - 0.22, x + 0.22], [mean, mean], color="#111111", linewidth=1.2)
        ax_a.text(x, mean - 0.9, f"mean {mean:.2f}", ha="center", va="top", fontsize=6)
    ax_a.axhline(0, color="#333333", linewidth=0.8)
    ax_a.set_xticks(list(positions.values()))
    ax_a.set_xticklabels(["3", "7", "14"])
    ax_a.set_xlabel("Target data budget (days)")
    ax_a.set_ylabel("MAE delta: SRFT - pretrain+FT")
    ax_a.set_title("Source replay improves DLinear in every building")
    ax_a.grid(axis="y", alpha=0.22, linewidth=0.5)
    panel_label(ax_a, "a")

    heat = wide.pivot(index="target", columns="k", values="delta_srft_vs_m3")
    heat = heat.loc[heat.mean(axis=1).sort_values().index]
    vmax = max(abs(float(heat.min().min())), abs(float(heat.max().max())))
    im = ax_b.imshow(heat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_b.set_xticks(np.arange(heat.shape[1]))
    ax_b.set_xticklabels([str(c) for c in heat.columns])
    ax_b.set_yticks(np.arange(heat.shape[0]))
    ax_b.set_yticklabels(heat.index, fontsize=5.8)
    ax_b.set_xlabel("Target data budget (days)")
    ax_b.set_title("Building-level MAE deltas")
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
    cbar.set_label("MAE delta", rotation=270, labelpad=10)
    panel_label(ax_b, "b")

    fig.tight_layout(pad=1.1)
    save_pub(fig, "figure2_srft_building_effect", 7.2, 4.8)
    plt.close(fig)


def plot_figure3(selected: pd.DataFrame):
    """Compact final leaderboard for the manuscript."""
    summary = selected.groupby(["method", "method_label", "k"])["MAE"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["n"])
    summary = summary[summary["method"].isin(METHOD_ORDER)]
    summary.to_csv(OUT / "source_data_figure3_leaderboard.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.0), sharey=True)
    display_order = [
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "dlinear_M3R_lambda0.2",
        "dlinear_M3_pretrain_ft",
        "lstm_M3_pretrain_ft",
        "dlinear_M2_target_only",
        "lstm_M2_target_only",
    ]
    labels = [LABELS[m] for m in display_order]
    for ax, k in zip(axes, [3, 7, 14]):
        sub = summary[summary["k"] == k].set_index("method").reindex(display_order)
        y = np.arange(len(sub))
        ax.barh(y, sub["mean"], xerr=sub["se"],
                color=[PALETTE[m] for m in display_order], capsize=2.4, height=0.72)
        ax.set_title(f"k={k} days")
        ax.set_xlabel("MAE")
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.22, linewidth=0.5)
    panel_label(axes[0], "a")
    fig.tight_layout(pad=1.1)
    save_pub(fig, "figure3_final_leaderboard", 7.2, 4.0)
    plt.close(fig)


def write_figure_notes():
    notes = """# Nature-Style Figure Package

Backend: Python/matplotlib only.

Figure 1 conclusion:
DLinear-SRFT becomes competitive with strong source+target baselines as the target data budget increases, while k=3 remains dominated by simple/classical baselines.

Figure 2 conclusion:
Source replay consistently improves DLinear over standard pretrain+fine-tune across target buildings.

Figure 3 conclusion:
The final method leaderboard should be interpreted conditionally by target-data budget; the proposed method is not universally best.

Exports:
Each figure is saved as SVG, PDF, PNG, and TIFF. Source-data CSV files are saved in the same directory.
"""
    (OUT / "figure_package_notes.md").write_text(notes, encoding="utf-8")


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    selected, building, _ = load_data()
    plot_figure1(selected, building)
    plot_figure2(building)
    plot_figure3(selected)
    write_figure_notes()
    print(f"[nature-figures] saved to {OUT}")


if __name__ == "__main__":
    main()
