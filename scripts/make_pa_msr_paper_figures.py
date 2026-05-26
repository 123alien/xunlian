#!/usr/bin/env python
"""Create PA-MSR paper figures from final claim evidence tables.

The figure set is intentionally tied to the final PA-MSR/PA-MSR+ claim,
not to the earlier PARBoost/DSOF exploration.
"""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / "results" / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "results" / "final_claim_evidence"
BDG2_120 = ROOT / "results" / "pa_msr_bdg2_120_light_cached_summary"
OUT = ROOT / "figures" / "pa_msr"
SRC = OUT / "source_data"
K_ORDER = [3, 7, 14, 30]

LABEL = {
    "M9_REVIN_MSR": "PA-MSR",
    "M11_REVIN_MSR_DPS_CAL": "PA-MSR+",
    "M3_ANCHOR_REVIN": "Residual RevIN",
    "PARBoost": "Prior residual\nboosting",
    "Persistence": "Persistence",
    "RF-ST": "RF-ST",
    "ExtraTrees-ST": "ExtraTrees-ST",
    "HistGBDT-ST": "HistGBDT-ST",
    "Seasonal naive": "Seasonal naive",
    "DLinear-SRFT": "DLinear-SRFT",
    "DLinear-M3": "DLinear-M3",
    "PA-SRFT": "PA-SRFT",
}

COLOR = {
    "PA-MSR": "#1F6F8B",
    "PA-MSR+": "#7A4E9D",
    "Residual RevIN": "#93B7C9",
    "Prior residual\nboosting": "#B7A57A",
    "Prior residual boosting": "#B7A57A",
    "Persistence": "#6F6F6F",
    "RF-ST": "#4F8A5B",
    "ExtraTrees-ST": "#8BAE5A",
    "HistGBDT-ST": "#C45A4A",
    "Seasonal naive": "#CFCFCF",
    "DLinear-SRFT": "#D17B35",
    "DLinear-M3": "#E4A15E",
    "PA-SRFT": "#D6B38A",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_fig(fig, stem: str, source: pd.DataFrame | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(
            OUT / f"{stem}.{ext}",
            bbox_inches="tight",
            dpi=600 if ext in {"png", "tiff"} else None,
        )
    if source is not None:
        source.to_csv(SRC / f"{stem}_source_data.csv", index=False)
    plt.close(fig)


def add_panel(ax, text: str) -> None:
    ax.text(
        -0.13,
        1.08,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def box(ax, xy, w, h, text, fc, ec, fontsize=6.4):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.75,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#555555", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def figure1_protocol() -> None:
    """Schematic-led figure: no-test-leakage cold-start protocol."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#222222"
    muted = "#6D6D6D"
    red = "#B84B4B"
    green = "#6F9E6E"
    blue = "#5E83A6"
    violet = "#8A6BA7"
    amber = "#B89A54"

    ax.text(0.05, 0.94, "Cold-start data flow", fontsize=9, fontweight="bold", ha="left", color=ink)
    ax.text(
        0.05,
        0.895,
        "Source selection, normalization, fitting, and calibration are completed before held-out testing.",
        fontsize=6.2,
        ha="left",
        color=muted,
    )

    x0, y0, h, total = 0.12, 0.72, 0.06, 0.78
    train, val, test = total * 0.60, total * 0.20, total * 0.20
    adapt = train * 0.25
    blocks = [
        (x0, adapt, "#EEF7EE", green, "k-day\nadapt"),
        (x0 + adapt, train - adapt, "#F2F2F2", "#9C9C9C", "target train tail"),
        (x0 + train, val, "#F7F0DE", amber, "validation"),
        (x0 + train + val, test, "#F9E8E6", red, "test only"),
    ]
    for xb, wb, fc, ec, lab in blocks:
        ax.add_patch(Rectangle((xb, y0), wb, h, facecolor=fc, edgecolor=ec, linewidth=0.7))
        ax.text(xb + wb / 2, y0 + h / 2, lab, ha="center", va="center", fontsize=5.5, color=ec if ec == red else ink)

    barrier = x0 + train + val
    ax.plot([barrier, barrier], [0.24, 0.83], color=red, lw=0.9, ls=(0, (3, 2.4)))
    ax.text(barrier - 0.01, 0.84, "no test leakage", fontsize=5.8, color=red, ha="right")

    ax.text(0.10, 0.54, "Source buildings", fontsize=6.4, fontweight="bold", ha="left", color=ink)
    for i, xx in enumerate([0.11, 0.14, 0.17, 0.20]):
        ax.add_patch(Rectangle((xx, 0.46), 0.02, 0.055, facecolor="#EEF3F8", edgecolor=blue, lw=0.65))
        ax.add_patch(Rectangle((xx + 0.005, 0.49), 0.004, 0.008, facecolor=blue, edgecolor="none", alpha=0.7))
        ax.add_patch(Rectangle((xx + 0.012, 0.475), 0.004, 0.008, facecolor=blue, edgecolor="none", alpha=0.7))

    box(ax, (0.30, 0.445), 0.135, 0.095, "Residual\nRevIN", "#EDF5F8", blue)
    box(ax, (0.48, 0.445), 0.145, 0.095, "Multi-scale\nresiduals", "#EDF5F8", blue)
    box(ax, (0.635, 0.445), 0.085, 0.095, "PA-MSR+\ncal.", "#F3EDF7", violet, fontsize=5.6)
    box(ax, (0.80, 0.445), 0.11, 0.095, "Held-out\nscoring", "#FFFFFF", red, fontsize=5.8)
    arrow(ax, (0.22, 0.49), (0.31, 0.49), blue)
    arrow(ax, (0.435, 0.49), (0.48, 0.49), blue)
    arrow(ax, (0.625, 0.49), (0.635, 0.49), violet)
    arrow(ax, (0.72, 0.49), (0.80, 0.49), red)
    arrow(ax, (x0 + adapt / 2, y0), (0.367, 0.54), green, rad=-0.04)
    arrow(ax, (x0 + train + val / 2, y0), (0.678, 0.54), amber, rad=0.05)
    arrow(ax, (x0 + train + val + test / 2, y0), (0.855, 0.54), red, rad=-0.08)

    ax.text(0.50, 0.27, r"$\hat{y}_{t+1}=y_t+s_i \hat{z}_{t+1}$", fontsize=9, ha="center", color=ink)
    ax.text(
        0.50,
        0.20,
        "Persistence defines the supervised residual target; validation only scales the residual amplitude in PA-MSR+.",
        fontsize=6.2,
        ha="center",
        color=muted,
    )

    src = pd.DataFrame(
        [
            {"item": "adaptation", "value": "first k days of target training period"},
            {"item": "source", "value": "non-target source-building training periods"},
            {"item": "validation", "value": "model selection and PA-MSR+ residual calibration only"},
            {"item": "test", "value": "held out until final evaluation"},
        ]
    )
    save_fig(fig, "fig1_protocol_no_leakage", src)


def figure2_leaderboard() -> None:
    """Quantitative grid: main leaderboard across datasets and k."""
    df = pd.read_csv(EVID / "table_final_leaderboard_ranked.csv")
    methods = [
        "M9_REVIN_MSR",
        "M11_REVIN_MSR_DPS_CAL",
        "M3_ANCHOR_REVIN",
        "PARBoost",
        "Persistence",
        "RF-ST",
        "ExtraTrees-ST",
        "HistGBDT-ST",
    ]
    df = df[df["method"].isin(methods)].copy()
    df["label"] = df["method"].map(LABEL)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    source_rows = []
    for ax, dataset, panel in zip(axes, ["BDG2-24", "COFACTOR-44"], ["a", "b"]):
        sub = df[df["dataset"] == dataset].copy()
        for method in methods:
            rows = sub[sub["method"] == method].sort_values("k")
            if rows.empty:
                continue
            lw = 2.2 if method in {"M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL"} else 1.1
            z = 5 if method in {"M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL"} else 2
            ax.plot(
                rows["k"],
                rows["mean_MAE"],
                marker="o",
                markersize=3.2,
                linewidth=lw,
                color=COLOR[LABEL[method]],
                label=LABEL[method],
                zorder=z,
            )
            source_rows.append(rows)
        ax.set_title(dataset)
        ax.set_xlabel("Target adaptation days (k)")
        ax.set_xticks(K_ORDER)
        ax.grid(axis="y", color="#D9D9D9", lw=0.55)
        add_panel(ax, panel)
    axes[0].set_ylabel("Mean MAE")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        fontsize=5.8,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.2)
    save_fig(fig, "fig2_main_leaderboard", pd.concat(source_rows, ignore_index=True))


def figure3_pairwise() -> None:
    """Quantitative grid: paired robustness and conditional module effect."""
    pair = pd.read_csv(EVID / "table_final_pairwise_robustness.csv")
    mod_b = pd.read_csv(EVID / "table_bdg2_24_module_pairwise.csv")
    mod_c = pd.read_csv(EVID / "table_cofactor_44_module_pairwise.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax_a, ax_b, ax_c = axes

    sub = pair[
        (pair["dataset"] == "COFACTOR-44")
        & (pair["candidate"] == "M9_REVIN_MSR")
        & (pair["baseline"].isin(["Persistence", "ExtraTrees-ST", "RF-ST"]))
    ].copy()
    for baseline, color in [
        ("Persistence", COLOR["Persistence"]),
        ("ExtraTrees-ST", COLOR["ExtraTrees-ST"]),
        ("RF-ST", COLOR["RF-ST"]),
    ]:
        rows = sub[sub["baseline"] == baseline].sort_values("k")
        ax_a.plot(rows["k"], rows["win_rate"] * 100, marker="o", lw=1.6, ms=3.2, color=color, label=f"vs {baseline}")
    ax_a.axhline(50, color="#B8B8B8", lw=0.8, ls="--")
    ax_a.set_ylim(0, 105)
    ax_a.set_xticks(K_ORDER)
    ax_a.set_title("COFACTOR paired wins")
    ax_a.set_xlabel("k days")
    ax_a.set_ylabel("PA-MSR win rate (%)")
    ax_a.legend(fontsize=5.5, loc="lower right")
    add_panel(ax_a, "a")

    sub = pair[
        (pair["dataset"] == "BDG2-24")
        & (pair["candidate"] == "M9_REVIN_MSR")
        & (pair["baseline"].isin(["M3_ANCHOR_REVIN", "Persistence", "ExtraTrees-ST"]))
    ].copy()
    for baseline, color in [
        ("M3_ANCHOR_REVIN", COLOR["Residual RevIN"]),
        ("Persistence", COLOR["Persistence"]),
        ("ExtraTrees-ST", COLOR["ExtraTrees-ST"]),
    ]:
        rows = sub[sub["baseline"] == baseline].sort_values("k")
        ax_b.plot(rows["k"], rows["negative_transfer_rate"] * 100, marker="o", lw=1.6, ms=3.2, color=color, label=f"vs {LABEL.get(baseline, baseline)}")
    ax_b.set_ylim(0, 55)
    ax_b.set_xticks(K_ORDER)
    ax_b.set_title("BDG2 negative transfer")
    ax_b.set_xlabel("k days")
    ax_b.set_ylabel("Rate (%)")
    ax_b.legend(fontsize=5.2, loc="upper right")
    add_panel(ax_b, "b")

    sub = pd.concat([mod_b, mod_c], ignore_index=True)
    sub = sub[
        (sub["candidate"] == "M11_REVIN_MSR_DPS_CAL")
        & (sub["baseline"] == "M9_REVIN_MSR")
    ].copy()
    width = 0.35
    x = np.arange(len(K_ORDER))
    for offset, dataset, color in [
        (-width / 2, "BDG2-24", COLOR["PA-MSR"]),
        (width / 2, "COFACTOR-44", COLOR["PA-MSR+"]),
    ]:
        rows = sub[sub["dataset"] == dataset].set_index("k").loc[K_ORDER]
        ax_c.bar(x + offset, rows["mean_improvement_MAE"], width, color=color, label=dataset)
    ax_c.axhline(0, color="#666666", lw=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(k) for k in K_ORDER])
    ax_c.set_title("PA-MSR+ vs PA-MSR")
    ax_c.set_xlabel("k days")
    ax_c.set_ylabel("Mean MAE improvement")
    ax_c.legend(fontsize=5.6)
    add_panel(ax_c, "c")

    for ax in axes:
        ax.grid(axis="y", color="#E1E1E1", lw=0.5)
    fig.tight_layout(w_pad=1.1)
    save_fig(fig, "fig3_pairwise_robustness", pd.concat([pair, mod_b, mod_c], ignore_index=True, sort=False))


def figure4_active_inactive() -> None:
    """Quantitative grid: BDG2 active/inactive sensitivity."""
    df = pd.read_csv(EVID / "table_bdg2_24_active_inactive_msr.csv")
    methods = ["M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL", "M3_ANCHOR_REVIN"]
    df = df[df["method"].isin(methods)].copy()
    df["label"] = df["method"].map(LABEL)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for ax, group, panel in zip(axes, ["active", "inactive_or_zero"], ["a", "b"]):
        sub = df[df["activity_group"] == group]
        for method in methods:
            rows = sub[sub["method"] == method].sort_values("k")
            ax.plot(
                rows["k"],
                rows["mean_MAE"],
                marker="o",
                lw=2.0 if method == "M9_REVIN_MSR" else 1.4,
                ms=3.2,
                color=COLOR[LABEL[method]],
                label=LABEL[method],
            )
        ax.set_xticks(K_ORDER)
        ax.set_xlabel("Target adaptation days (k)")
        ax.set_title("Active targets" if group == "active" else "Inactive/near-zero targets")
        ax.grid(axis="y", color="#E1E1E1", lw=0.5)
        add_panel(ax, panel)
    axes[0].set_ylabel("Mean MAE")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean MAE (log scale)")
    axes[1].legend(fontsize=5.8, loc="upper right")
    fig.tight_layout(w_pad=1.1)
    save_fig(fig, "fig4_bdg2_active_inactive", df)


def supplementary_bdg2_120_scale_validation() -> None:
    """Supplementary figure: PA-MSR scale validation on BDG2-120."""
    leader_path = BDG2_120 / "table_pa_msr_bdg2_120_leaderboard.csv"
    pair_path = BDG2_120 / "table_pa_msr_bdg2_120_pairwise.csv"
    if not leader_path.exists() or not pair_path.exists():
        print(f"Skipped BDG2-120 supplementary figure; missing {BDG2_120}")
        return

    leader = pd.read_csv(leader_path)
    pair = pd.read_csv(pair_path)
    methods = ["PA-MSR", "Prior residual boosting", "Persistence", "RF-ST", "ExtraTrees-ST"]
    leader = leader[leader["method_label"].isin(methods)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    ax_a, ax_b = axes
    for method in methods:
        rows = leader[leader["method_label"] == method].sort_values("k")
        if rows.empty:
            continue
        lw = 2.2 if method == "PA-MSR" else 1.15
        z = 5 if method == "PA-MSR" else 2
        ax_a.plot(
            rows["k"],
            rows["mean_MAE"],
            marker="o",
            markersize=3.2,
            linewidth=lw,
            color=COLOR.get(method, "#888888"),
            label=method,
            zorder=z,
        )
    ax_a.set_title("BDG2-120 active robustness")
    ax_a.set_xlabel("Target adaptation days (k)")
    ax_a.set_ylabel("Mean MAE")
    ax_a.set_xticks(K_ORDER)
    ax_a.grid(axis="y", color="#D9D9D9", lw=0.55)
    add_panel(ax_a, "a")

    baselines = ["Persistence", "RF-ST", "ExtraTrees-ST", "Prior residual boosting"]
    for baseline in baselines:
        rows = pair[pair["baseline"] == baseline].sort_values("k")
        if rows.empty:
            continue
        ax_b.plot(
            rows["k"],
            rows["win_rate"] * 100,
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            color=COLOR.get(baseline, "#888888"),
            label=f"vs {baseline}",
        )
    ax_b.axhline(50, color="#B8B8B8", lw=0.8, ls="--")
    ax_b.set_ylim(0, 105)
    ax_b.set_title("PA-MSR paired wins")
    ax_b.set_xlabel("Target adaptation days (k)")
    ax_b.set_ylabel("Win rate (%)")
    ax_b.set_xticks(K_ORDER)
    ax_b.grid(axis="y", color="#D9D9D9", lw=0.55)
    add_panel(ax_b, "b")

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    fig.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=5.8,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=1.2)
    source = pd.concat(
        [
            leader.assign(panel="a_leaderboard"),
            pair[pair["baseline"].isin(baselines)].assign(panel="b_pairwise"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_fig(fig, "figS1_bdg2_120_pa_msr_scale_validation", source)


def main() -> None:
    setup_style()
    figure1_protocol()
    figure2_leaderboard()
    figure3_pairwise()
    figure4_active_inactive()
    supplementary_bdg2_120_scale_validation()
    print(f"Saved PA-MSR figures to {OUT}")


if __name__ == "__main__":
    main()
