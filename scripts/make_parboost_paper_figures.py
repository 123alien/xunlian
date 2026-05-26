#!/usr/bin/env python
"""Create paper-facing PARBoost figures from generated evidence tables."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "parboost"
SRC = OUT / "source_data"

K_ORDER = [3, 7, 14, 30]
METHOD_LABELS = {
    "PAR_HIST_GBDT_SW_CAL": "PARBoost",
    "persistence": "Persistence",
    "random_forest_source_target": "RF-ST",
    "extra_trees_source_target": "ExtraTrees-ST",
    "hist_gbdt_source_target": "HistGBDT-ST",
    "seasonal_naive_24": "Seasonal naive",
    "ABL_DIRECT_HISTGBDT_ST": "Direct HistGBDT-ST",
    "ABL_RESIDUAL_UNIFORM": "Residual",
    "ABL_RESIDUAL_SIM": "Residual + similarity",
    "ABL_RESIDUAL_CAL": "Residual + calibration",
}
COLORS = {
    "PARBoost": "#0F4D92",
    "Persistence": "#767676",
    "RF-ST": "#9A4D8E",
    "ExtraTrees-ST": "#42949E",
    "HistGBDT-ST": "#B64342",
    "Seasonal naive": "#CFCECE",
    "Direct HistGBDT-ST": "#B64342",
    "Residual": "#B4C0E4",
    "Residual + similarity": "#7884B4",
    "Residual + calibration": "#E4CCD8",
}


def setup_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save_fig(fig, stem: str, source_data: pd.DataFrame | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    if source_data is not None:
        source_data.to_csv(SRC / f"{stem}_source_data.csv", index=False)
    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def rounded_box(ax, xy, width, height, text, fc="#FFFFFF", ec="#4D4D4D",
                lw=0.9, fontsize=7, color="#222222", radius=0.04):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        linespacing=1.15,
    )
    return box


def arrow(ax, start, end, color="#4D4D4D", lw=1.0, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def load_tail():
    return pd.read_csv(ROOT / "results" / "parboost_evidence" / "table_tail_risk_by_dataset_method.csv")


def load_pairwise():
    return pd.read_csv(ROOT / "results" / "parboost_evidence" / "table_paired_tests_parboost.csv")


def figure_0_protocol_schematic():
    """Schematic evidence: strict cold-start data flow without test leakage."""
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#4D6F91"
    teal = "#6C9A9E"
    green = "#7EA27A"
    amber = "#B69A55"
    red = "#B85C58"
    ink = "#2A2A2A"
    muted = "#707070"
    pale_blue = "#EEF3F8"
    pale_teal = "#EFF7F7"
    pale_green = "#EFF6EE"
    pale_amber = "#F8F1DF"
    pale_red = "#F9E8E6"
    pale_gray = "#F4F4F4"

    ax.text(0.035, 0.94, "a", fontsize=9, fontweight="bold", ha="left", va="top", color=ink)
    ax.text(0.065, 0.94, "Cold-start evaluation protocol", fontsize=8.5, fontweight="bold", ha="left", va="top", color=ink)
    ax.text(0.065, 0.895, "Target test data are held out from every upstream operation.", fontsize=6.4, ha="left", va="top", color=muted)

    # Target timeline as the hero: a thin, journal-style band.
    x0, y0, h, total_w = 0.12, 0.72, 0.058, 0.78
    train_w, val_w, test_w = total_w * 0.60, total_w * 0.20, total_w * 0.20
    adapt_w = train_w * 0.26
    blocks = [
        (x0, adapt_w, pale_green, green, "k-day\nadapt"),
        (x0 + adapt_w, train_w - adapt_w, pale_gray, "#9B9B9B", "target train tail"),
        (x0 + train_w, val_w, pale_amber, amber, "validation"),
        (x0 + train_w + val_w, test_w, pale_red, red, "test only"),
    ]
    for xb, wb, fc, ec, label in blocks:
        ax.add_patch(Rectangle((xb, y0), wb, h, facecolor=fc, edgecolor=ec, linewidth=0.65))
        ax.text(xb + wb / 2, y0 + h / 2, label, fontsize=5.5, ha="center", va="center", color=ink if ec != red else red)
    ax.text(x0, y0 + h + 0.035, "Target timeline", fontsize=6.8, fontweight="bold", ha="left", color=ink)
    ax.text(x0 + train_w / 2, y0 - 0.025, "train", fontsize=5.5, ha="center", color=muted)
    ax.text(x0 + train_w + val_w / 2, y0 - 0.025, "validation", fontsize=5.5, ha="center", color=muted)
    ax.text(x0 + train_w + val_w + test_w / 2, y0 - 0.025, "held-out test", fontsize=5.5, ha="center", color=red)

    barrier_x = x0 + train_w + val_w
    ax.axvspan(barrier_x, x0 + total_w, ymin=0.18, ymax=0.82, facecolor=red, alpha=0.045, edgecolor=None)
    ax.plot([barrier_x, barrier_x], [0.25, 0.83], color=red, lw=0.9, ls=(0, (3.5, 2.7)))
    ax.text(barrier_x - 0.01, 0.84, "no test data upstream", fontsize=5.7, color=red, ha="right", va="bottom")

    # Minimal source pool icons.
    ax.text(0.12, 0.555, "Source pool", fontsize=6.4, fontweight="bold", ha="left", color=ink)
    icon_y = 0.49
    for i, xx in enumerate([0.125, 0.155, 0.185, 0.215]):
        ax.add_patch(Rectangle((xx, icon_y), 0.018, 0.05, facecolor=pale_blue, edgecolor=blue, linewidth=0.65))
        ax.add_patch(Rectangle((xx + 0.004, icon_y + 0.032), 0.004, 0.006, facecolor=blue, edgecolor="none", alpha=0.55))
        ax.add_patch(Rectangle((xx + 0.011, icon_y + 0.020), 0.004, 0.006, facecolor=blue, edgecolor="none", alpha=0.55))
    ax.text(0.12, 0.455, "target excluded", fontsize=5.4, ha="left", color=muted)

    # Streamlined pipeline.
    rounded_box(ax, (0.32, 0.455), 0.16, 0.09, "Similarity\n+ scaling", fc=pale_teal, ec=teal, fontsize=6.1, radius=0.018)
    rounded_box(ax, (0.55, 0.455), 0.17, 0.09, "Residual\nlearner", fc=pale_blue, ec=blue, fontsize=6.1, radius=0.018)
    rounded_box(ax, (0.79, 0.455), 0.13, 0.09, "Held-out\nscoring", fc="#FFFFFF", ec=red, fontsize=6.1, radius=0.018)

    arrow(ax, (0.24, 0.50), (0.32, 0.50), color=blue, lw=0.75)
    arrow(ax, (0.48, 0.50), (0.55, 0.50), color=muted, lw=0.75)
    arrow(ax, (0.72, 0.50), (0.79, 0.50), color=muted, lw=0.75)
    arrow(ax, (x0 + adapt_w / 2, y0), (0.40, 0.545), color=green, lw=0.75, rad=-0.05)
    arrow(ax, (x0 + train_w + val_w / 2, y0), (0.635, 0.545), color=amber, lw=0.75, rad=0.05)
    arrow(ax, (x0 + train_w + val_w + test_w / 2, y0), (0.855, 0.545), color=red, lw=0.75, rad=-0.06)

    ax.text(0.40, 0.405, "source train + k days", fontsize=5.3, ha="center", color=muted)
    ax.text(0.635, 0.405, "fit residual targets", fontsize=5.3, ha="center", color=muted)
    ax.text(0.855, 0.395, "evaluation only", fontsize=5.3, ha="center", color=red)

    # Footer strip: short and unobtrusive.
    ax.plot([0.16, 0.84], [0.275, 0.275], color="#BDBDBD", lw=0.65)
    ax.text(0.50, 0.225, "forecast = last load + calibrated residual correction", fontsize=6.5, ha="center", color=ink)
    ax.text(0.50, 0.177, "Persistence defines the residual target; validation only scales the residual amplitude.", fontsize=5.8, ha="center", color=muted)

    source = pd.DataFrame(
        [
            {"item": "source_policy", "value": "all non-target buildings, training period only"},
            {"item": "target_adaptation", "value": "first k days of target training period"},
            {"item": "validation_use", "value": "residual amplitude calibration and model selection only"},
            {"item": "test_use", "value": "final evaluation only"},
            {"item": "leakage_rule", "value": "test data not used for source selection, scaling, fitting, or calibration"},
        ]
    )
    save_fig(fig, "fig1_protocol_schematic", source)


def figure_1_benchmark_summary():
    """Benchmark evidence: BDG2 robust gains plus COFACTOR boundary."""
    tail = load_tail()
    pair = load_pairwise()
    methods = [
        "PAR_HIST_GBDT_SW_CAL",
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
    ]
    labels = [METHOD_LABELS[m] for m in methods]

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 1], wspace=0.42, hspace=0.48)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    bdg = tail[(tail["dataset"] == "BDG2-120") & (tail["method"].isin(methods))]
    pivot = bdg.pivot(index="k", columns="method", values="mean_MAE").loc[K_ORDER, methods]
    x = np.arange(len(K_ORDER))
    for m, lab in zip(methods, labels):
        ax_a.plot(x, pivot[m], marker="o", lw=1.5, ms=3.5, color=COLORS[lab], label=lab)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(k) for k in K_ORDER])
    ax_a.set_xlabel("Target adaptation days (k)")
    ax_a.set_ylabel("Mean MAE")
    ax_a.set_title("BDG2-120 active robustness")
    ax_a.legend(loc="upper right", fontsize=6)
    add_panel_label(ax_a, "a")

    bdg_pair = pair[(pair["dataset"] == "BDG2-120") & (pair["baseline"].isin(["persistence", "random_forest_source_target", "extra_trees_source_target"]))]
    win_piv = bdg_pair.pivot(index="k", columns="baseline", values="win_rate").loc[K_ORDER]
    for baseline, color in [
        ("persistence", COLORS["Persistence"]),
        ("random_forest_source_target", COLORS["RF-ST"]),
        ("extra_trees_source_target", COLORS["ExtraTrees-ST"]),
    ]:
        ax_b.plot(x, win_piv[baseline] * 100, marker="o", lw=1.4, ms=3.2, color=color, label=METHOD_LABELS[baseline])
    ax_b.axhline(50, color="#B8B8B8", lw=0.8, ls="--")
    ax_b.set_ylim(0, 105)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(k) for k in K_ORDER])
    ax_b.set_ylabel("PARBoost win rate (%)")
    ax_b.set_title("Paired wins on BDG2-120")
    ax_b.legend(fontsize=5.8, loc="lower right")
    add_panel_label(ax_b, "b")

    p90 = bdg.pivot(index="k", columns="method", values="p90_MAE").loc[K_ORDER, methods]
    for m, lab in zip(methods, labels):
        ax_c.plot(x, p90[m], marker="o", lw=1.4, ms=3.2, color=COLORS[lab], label=lab)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(k) for k in K_ORDER])
    ax_c.set_ylabel("P90 MAE")
    ax_c.set_title("BDG2 tail risk")
    add_panel_label(ax_c, "c")

    cof = tail[(tail["dataset"] == "COFACTOR-44") & (tail["method"].isin(methods))]
    cof_piv = cof.pivot(index="k", columns="method", values="mean_MAE").loc[K_ORDER, methods]
    for m, lab in zip(methods, labels):
        ax_d.plot(x, cof_piv[m], marker="o", lw=1.4, ms=3.2, color=COLORS[lab], label=lab)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([str(k) for k in K_ORDER])
    ax_d.set_xlabel("Target adaptation days (k)")
    ax_d.set_ylabel("Mean MAE")
    ax_d.set_title("COFACTOR-44 external boundary")
    add_panel_label(ax_d, "d")

    cof_pair = pair[(pair["dataset"] == "COFACTOR-44") & (pair["baseline"].isin(["persistence", "extra_trees_source_target"]))]
    width = 0.34
    p_vals = cof_pair[cof_pair["baseline"] == "persistence"].set_index("k").loc[K_ORDER, "win_rate"].to_numpy() * 100
    e_vals = cof_pair[cof_pair["baseline"] == "extra_trees_source_target"].set_index("k").loc[K_ORDER, "win_rate"].to_numpy() * 100
    ax_e.bar(x - width / 2, p_vals, width, color=COLORS["Persistence"], label="vs Persistence")
    ax_e.bar(x + width / 2, e_vals, width, color=COLORS["ExtraTrees-ST"], label="vs ExtraTrees-ST")
    ax_e.axhline(50, color="#B8B8B8", lw=0.8, ls="--")
    ax_e.set_ylim(0, 105)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([str(k) for k in K_ORDER])
    ax_e.set_xlabel("Target adaptation days (k)")
    ax_e.set_ylabel("PARBoost win rate (%)")
    ax_e.set_title("COFACTOR paired wins")
    ax_e.legend(fontsize=5.8, loc="upper right")
    add_panel_label(ax_e, "e")

    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
        ax.tick_params(labelsize=6.5)
    source = pd.concat(
        [
            bdg.assign(panel="a/c"),
            bdg_pair.assign(panel="b"),
            cof.assign(panel="d"),
            cof_pair.assign(panel="e"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_fig(fig, "fig1_benchmark_summary", source)


def figure_2_mechanism_ablation():
    """Mechanism evidence: residual anchoring is the dominant gain."""
    ab = pd.read_csv(ROOT / "results" / "parboost_ablation_bdg2" / "table_par_gbdt_pilot.csv")
    methods = [
        "ABL_DIRECT_HISTGBDT_ST",
        "ABL_RESIDUAL_UNIFORM",
        "ABL_RESIDUAL_SIM",
        "ABL_RESIDUAL_CAL",
        "PAR_HIST_GBDT_SW_CAL",
    ]
    ab = ab[ab["method"].isin(methods)].copy()
    building_level = ab.groupby(["target", "k", "method"], as_index=False)["MAE"].mean()
    summary = building_level.groupby(["k", "method"], as_index=False).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        p90_MAE=("MAE", lambda s: s.quantile(0.90)),
        worst_MAE=("MAE", "max"),
    )

    labels = [METHOD_LABELS[m] for m in methods]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax_a, ax_b, ax_c = axes
    x = np.arange(len(K_ORDER))

    for m, lab in zip(methods, labels):
        y = summary[summary["method"] == m].set_index("k").loc[K_ORDER, "mean_MAE"].to_numpy()
        ax_a.plot(x, y, marker="o", lw=1.4, ms=3.2, color=COLORS.get(lab, "#333333"), label=lab)
    ax_a.set_yscale("log")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(k) for k in K_ORDER])
    ax_a.set_xlabel("Target adaptation days (k)")
    ax_a.set_ylabel("Mean MAE (log scale)")
    ax_a.set_title("Direct load prediction fails")
    ax_a.legend(fontsize=5.5, loc="upper right")
    add_panel_label(ax_a, "a")

    k_show = 14
    sub = building_level[building_level["k"] == k_show].pivot(index="target", columns="method", values="MAE").dropna()
    delta = sub["PAR_HIST_GBDT_SW_CAL"] - sub["ABL_DIRECT_HISTGBDT_ST"]
    ax_b.axvline(0, color="#B8B8B8", lw=0.8, ls="--")
    y = np.arange(len(delta))
    delta_sorted = delta.sort_values()
    ax_b.scatter(delta_sorted.values, y, s=12, color="#0F4D92", edgecolor="white", linewidth=0.3)
    ax_b.set_yticks([])
    ax_b.set_xlabel("PARBoost - direct MAE")
    ax_b.set_title(f"Paired building deltas, k={k_show}")
    add_panel_label(ax_b, "b")

    p90 = summary.pivot(index="k", columns="method", values="p90_MAE").loc[K_ORDER, methods]
    bars = [
        p90["ABL_DIRECT_HISTGBDT_ST"].to_numpy(),
        p90["PAR_HIST_GBDT_SW_CAL"].to_numpy(),
    ]
    width = 0.36
    ax_c.bar(x - width / 2, bars[0], width, color=COLORS["Direct HistGBDT-ST"], label="Direct")
    ax_c.bar(x + width / 2, bars[1], width, color=COLORS["PARBoost"], label="PARBoost")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(k) for k in K_ORDER])
    ax_c.set_xlabel("Target adaptation days (k)")
    ax_c.set_ylabel("P90 MAE")
    ax_c.set_title("Tail risk after anchoring")
    ax_c.legend(fontsize=5.8)
    add_panel_label(ax_c, "c")

    for ax in axes:
        ax.tick_params(labelsize=6.5)
    save_fig(fig, "fig2_mechanism_ablation", summary.merge(building_level, on=["k", "method"], how="right"))


def figure_3_stratified_persistence():
    """Stratified evidence: where the persistence anchor helps."""
    strat = pd.read_csv(ROOT / "results" / "parboost_evidence" / "table_stratified_by_persistence_strength.csv")
    strat = strat[strat["dataset"].isin(["BDG2-120", "COFACTOR-44"])].copy()
    order = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), sharey=False)
    for ax, dataset, label in zip(axes, ["BDG2-120", "COFACTOR-44"], ["a", "b"]):
        sub = strat[strat["dataset"] == dataset].copy()
        width = 0.2
        x = np.arange(len(order))
        for i, k in enumerate(K_ORDER):
            vals = sub[sub["k"] == k].set_index("stratum").loc[order, "mean_delta_vs_persistence"].to_numpy()
            ax.bar(x + (i - 1.5) * width, vals, width, label=f"k={k}", color=["#B4C0E4", "#7884B4", "#E4CCD8", "#0F4D92"][i])
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("Persistence-MAE stratum")
        ax.set_ylabel("Mean MAE delta vs Persistence")
        ax.set_title(dataset)
        ax.legend(fontsize=5.8, ncol=2)
        add_panel_label(ax, label)
        ax.tick_params(labelsize=6.5)
    save_fig(fig, "fig3_stratified_persistence", strat)


def main():
    setup_style()
    figure_0_protocol_schematic()
    figure_1_benchmark_summary()
    figure_2_mechanism_ablation()
    figure_3_stratified_persistence()
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
