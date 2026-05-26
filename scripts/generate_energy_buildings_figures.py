#!/usr/bin/env python
"""Clean Energy and Buildings style figures for the forecasting manuscript.

Design goals:
- restrained engineering-journal style
- readable when printed in grayscale
- simple panels with direct evidence
- PDF/SVG/TIFF/PNG exports plus source data
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("results/phase2_paper/energy_buildings_figures")
PAPER = Path("results/phase2_paper/forecasting_paper")


METHODS_MAIN = [
    "persistence",
    "random_forest_source_target",
    "extra_trees_source_target",
    "dlinear_M3R_lambda0.2",
    "dlinear_M3_pretrain_ft",
    "lstm_M3_pretrain_ft",
]

METHODS_FULL = [
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

COLORS = {
    "persistence": "#4D4D4D",
    "random_forest_source_target": "#2E6F40",
    "extra_trees_source_target": "#7A9E46",
    "dlinear_M2_target_only": "#A6CEE3",
    "dlinear_M3_pretrain_ft": "#4F8CC9",
    "dlinear_M3R_lambda0.2": "#0B3C73",
    "lstm_M2_target_only": "#F4B183",
    "lstm_M3_pretrain_ft": "#C55A11",
}

MARKERS = {
    "persistence": "o",
    "random_forest_source_target": "s",
    "extra_trees_source_target": "^",
    "dlinear_M3R_lambda0.2": "D",
    "dlinear_M3_pretrain_ft": "P",
    "lstm_M3_pretrain_ft": "X",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(
            OUT / f"{stem}.{ext}",
            bbox_inches="tight",
            dpi=600 if ext in {"png", "tiff"} else None,
        )


def load_data():
    all_methods = pd.read_csv(PAPER / "table_all_forecasting_methods.csv")
    building = pd.read_csv(PAPER / "table_all_forecasting_methods_building_level.csv")
    return all_methods, building


def summarize_runs(df: pd.DataFrame, methods: list[str], metric: str) -> pd.DataFrame:
    sub = df[df["method"].isin(methods)].copy()
    out = sub.groupby(["method", "k"])[metric].agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"])
    out["label"] = out["method"].map(LABELS)
    return out


def figure1_mae_curve(all_methods: pd.DataFrame):
    """Single clean line figure for main performance."""
    data = summarize_runs(all_methods, METHODS_MAIN, "MAE")
    data.to_csv(OUT / "source_data_eb_figure1_mae_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in METHODS_MAIN:
        sub = data[data["method"] == method].sort_values("k")
        lw = 2.4 if method == "dlinear_M3R_lambda0.2" else 1.6
        ax.errorbar(
            sub["k"], sub["mean"], yerr=sub["se"],
            marker=MARKERS.get(method, "o"),
            markersize=5,
            linewidth=lw,
            capsize=3,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set_xlabel("Available target-building data (days)")
    ax.set_ylabel("MAE")
    ax.set_xticks([3, 7, 14])
    ax.set_title("Cold-start forecasting performance")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(ncol=2, loc="upper right", handlelength=2.2)
    fig.tight_layout()
    save(fig, "eb_figure1_mae_curve")
    plt.close(fig)


def figure2_final_bar(all_methods: pd.DataFrame):
    """Grouped bar chart for final selected method comparison."""
    data = summarize_runs(all_methods, METHODS_FULL, "MAE")
    data.to_csv(OUT / "source_data_eb_figure2_method_bars.csv", index=False)

    display = [
        "Persistence",
        "RF source+target",
        "ExtraTrees source+target",
        "DLinear-SRFT",
        "DLinear pretrain+FT",
        "LSTM pretrain+FT",
        "DLinear target-only",
        "LSTM target-only",
    ]
    method_by_label = {v: k for k, v in LABELS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.6), sharey=True)
    for ax, k in zip(axes, [3, 7, 14]):
        rows = []
        for label in display:
            method = method_by_label[label]
            r = data[(data["method"] == method) & (data["k"] == k)].iloc[0]
            rows.append((label, method, r["mean"], r["se"]))
        y = np.arange(len(rows))
        ax.barh(
            y,
            [r[2] for r in rows],
            xerr=[r[3] for r in rows],
            color=[COLORS[r[1]] for r in rows],
            edgecolor="white",
            linewidth=0.5,
            capsize=2.5,
            height=0.72,
        )
        ax.set_title(f"k = {k} days")
        ax.set_xlabel("MAE")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    fig.suptitle("Comparison with simple, classical, and neural transfer baselines", y=1.02)
    fig.tight_layout()
    save(fig, "eb_figure2_method_bars")
    plt.close(fig)


def figure3_srft_delta(building: pd.DataFrame):
    """DLinear-SRFT building-level deltas in a clean dot plot."""
    wide = building[building["method"].isin([
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.2",
    ])].pivot_table(index=["target", "k"], columns="method", values="MAE").reset_index()
    wide["delta_MAE"] = wide["dlinear_M3R_lambda0.2"] - wide["dlinear_M3_pretrain_ft"]
    wide.to_csv(OUT / "source_data_eb_figure3_srft_delta.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    positions = {3: 0, 7: 1, 14: 2}
    rng = np.random.default_rng(11)
    for k, x in positions.items():
        vals = wide[wide["k"] == k]["delta_MAE"].to_numpy()
        jitter = rng.normal(0, 0.035, len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            s=30,
            color=COLORS["dlinear_M3R_lambda0.2"],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.4,
        )
        mean = vals.mean()
        ax.plot([x - 0.22, x + 0.22], [mean, mean], color="black", linewidth=1.4)
        ax.text(x, mean - 0.55, f"{mean:.2f}", ha="center", va="top", fontsize=8)
    ax.axhline(0, color="#4D4D4D", linewidth=0.9, linestyle="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["3", "7", "14"])
    ax.set_xlabel("Available target-building data (days)")
    ax.set_ylabel("MAE difference: DLinear-SRFT minus DLinear pretrain+FT")
    ax.set_title("Source replay improves DLinear transfer across buildings")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    save(fig, "eb_figure3_srft_delta")
    plt.close(fig)


def figure4_rank_heatmap(building: pd.DataFrame):
    """Optional simple grayscale top-3 heatmap."""
    b = building[building["method"].isin(METHODS_FULL)].copy()
    b["rank_MAE"] = b.groupby(["target", "k"])["MAE"].rank(method="min")
    top3 = b.groupby(["method", "k"])["rank_MAE"].apply(lambda x: np.mean(x <= 3)).reset_index(name="top3_rate")
    top3["label"] = top3["method"].map(LABELS)
    top3.to_csv(OUT / "source_data_eb_figure4_top3_heatmap.csv", index=False)

    order = [
        "DLinear-SRFT",
        "RF source+target",
        "ExtraTrees source+target",
        "Persistence",
        "DLinear pretrain+FT",
        "LSTM pretrain+FT",
        "DLinear target-only",
        "LSTM target-only",
    ]
    mat = top3.pivot(index="label", columns="k", values="top3_rate").reindex(order)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(mat.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns])
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel("Available target-building data (days)")
    ax.set_title("Top-3 rate across target buildings")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color="white" if v > 0.55 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Top-3 rate", rotation=270, labelpad=11)
    fig.tight_layout()
    save(fig, "eb_figure4_top3_heatmap_optional")
    plt.close(fig)


def write_notes():
    notes = """# Energy and Buildings Figure Package

Style:
- clean Elsevier engineering-journal style
- larger text than Nature-style package
- restrained colors and simple grids
- one central message per figure

Recommended main figures:
1. eb_figure1_mae_curve: main performance trend.
2. eb_figure2_method_bars: full selected-method comparison.
3. eb_figure3_srft_delta: DLinear-SRFT building-level improvement.

Optional supplementary:
4. eb_figure4_top3_heatmap_optional: top-3 rate by method and k.
"""
    (OUT / "energy_buildings_figure_notes.md").write_text(notes, encoding="utf-8")


def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    all_methods, building = load_data()
    figure1_mae_curve(all_methods)
    figure2_final_bar(all_methods)
    figure3_srft_delta(building)
    figure4_rank_heatmap(building)
    write_notes()
    print(f"[energy-buildings-figures] saved to {OUT}")


if __name__ == "__main__":
    main()
