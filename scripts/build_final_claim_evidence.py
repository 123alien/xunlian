#!/usr/bin/env python
"""Build final claim-evidence tables from completed module and baseline runs."""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - scipy may be absent in lightweight envs.
    wilcoxon = None


OUT = Path("results/final_claim_evidence")


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_delta(df: pd.DataFrame, candidate: str, baseline: str, dataset: str) -> pd.DataFrame:
    rows = []
    keys = ["target", "k"]
    for k in sorted(df["k"].unique()):
        a = df[(df["method"] == candidate) & (df["k"] == k)][keys + ["MAE"]].rename(columns={"MAE": "candidate_MAE"})
        b = df[(df["method"] == baseline) & (df["k"] == k)][keys + ["MAE"]].rename(columns={"MAE": "baseline_MAE"})
        m = a.merge(b, on=keys)
        if m.empty:
            continue
        delta = m["baseline_MAE"] - m["candidate_MAE"]
        rows.append({
            "dataset": dataset,
            "candidate": candidate,
            "baseline": baseline,
            "k": int(k),
            "n": int(len(m)),
            "mean_improvement_MAE": float(delta.mean()),
            "median_improvement_MAE": float(delta.median()),
            "win_rate": float((delta > 0).mean()),
            "negative_transfer_rate": float((delta < 0).mean()),
            "candidate_mean_MAE": float(m["candidate_MAE"].mean()),
            "baseline_mean_MAE": float(m["baseline_MAE"].mean()),
        })
    return pd.DataFrame(rows)


def summarize_module_with_active_split() -> None:
    mod = pd.read_csv("results/pa_module_screen_bdg2_24_msr_dps/aggregate/table_pa_module_extensions.csv")
    manifest = pd.read_csv("results/bdg2_full_eligible/table_eligibility_audit.csv")
    keep = ["building_id", "active_test", "test_zero_ratio", "test_persistence_mae", "building_type"]
    mod = mod.merge(manifest[keep], left_on="target", right_on="building_id", how="left")
    mod["activity_group"] = np.where(mod["active_test"].fillna(False), "active", "inactive_or_zero")

    focus = mod[mod["method"].isin(["M3_ANCHOR_REVIN", "M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL"])]
    split = focus.groupby(["activity_group", "method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        p90_MAE=("MAE", lambda x: float(np.percentile(x, 90))),
        worst_MAE=("MAE", "max"),
        n=("MAE", "size"),
    ).reset_index().sort_values(["activity_group", "k", "mean_MAE"])
    split.to_csv(OUT / "table_bdg2_24_active_inactive_msr.csv", index=False)
    write_markdown_table(split, OUT / "table_bdg2_24_active_inactive_msr.md")

    pairs = pd.concat([
        paired_delta(focus, "M9_REVIN_MSR", "M3_ANCHOR_REVIN", "BDG2-24"),
        paired_delta(focus, "M11_REVIN_MSR_DPS_CAL", "M3_ANCHOR_REVIN", "BDG2-24"),
        paired_delta(focus, "M11_REVIN_MSR_DPS_CAL", "M9_REVIN_MSR", "BDG2-24"),
    ], ignore_index=True)
    pairs.to_csv(OUT / "table_bdg2_24_module_pairwise.csv", index=False)
    write_markdown_table(pairs, OUT / "table_bdg2_24_module_pairwise.md")


def build_bdg2_candidate_main_table() -> None:
    old = pd.read_csv("results/parboost_evidence/manuscript_tables/table2_bdg2_24_full_leaderboard.csv")
    new = pd.read_csv("results/pa_module_screen_bdg2_24_msr_dps/aggregate/table_pa_module_extensions_summary.csv")
    new = new[new["method"].isin(["M3_ANCHOR_REVIN", "M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL"])].copy()
    new = new.rename(columns={"n": "n_buildings"})
    old = old.copy()
    old["n_buildings"] = 24
    cols = ["method", "k", "mean_MAE", "median_MAE", "worst_MAE", "n_buildings"]
    main = pd.concat([old[cols], new[cols]], ignore_index=True)
    main = main.sort_values(["k", "mean_MAE"])
    main.to_csv(OUT / "table_main_bdg2_24_with_msr_candidates.csv", index=False)
    write_markdown_table(main, OUT / "table_main_bdg2_24_with_msr_candidates.md")


def build_cofactor_candidate_main_table() -> None:
    old = pd.read_csv("results/parboost_evidence/manuscript_tables/table4_cofactor_44_external.csv")
    old = old.copy()
    old["n_buildings"] = 44
    old_keep = old[["method", "k", "mean_MAE", "median_MAE", "worst_MAE", "n_buildings"]]

    m3 = pd.read_csv("results/pa_module_extensions_cofactor_44/aggregate/table_pa_module_extensions_summary.csv")
    m3 = m3[m3["method"].isin(["M3_ANCHOR_REVIN"])].copy()
    m9 = pd.read_csv("results/pa_module_screen_cofactor_44_msr_dps/aggregate/table_pa_module_extensions_summary.csv")
    new = pd.concat([m3, m9], ignore_index=True)
    new = new.rename(columns={"n": "n_buildings"})
    new_keep = new[["method", "k", "mean_MAE", "median_MAE", "worst_MAE", "n_buildings"]]

    main = pd.concat([old_keep, new_keep], ignore_index=True)
    main = main.sort_values(["k", "mean_MAE"])
    main.to_csv(OUT / "table_main_cofactor_44_with_msr_candidates.csv", index=False)
    write_markdown_table(main, OUT / "table_main_cofactor_44_with_msr_candidates.md")

    full = pd.concat([
        pd.read_csv("results/pa_module_extensions_cofactor_44/aggregate/table_pa_module_extensions.csv"),
        pd.read_csv("results/pa_module_screen_cofactor_44_msr_dps/aggregate/table_pa_module_extensions.csv"),
    ], ignore_index=True)
    focus = full[full["method"].isin(["M3_ANCHOR_REVIN", "M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL"])]
    pairs = pd.concat([
        paired_delta(focus, "M9_REVIN_MSR", "M3_ANCHOR_REVIN", "COFACTOR-44"),
        paired_delta(focus, "M11_REVIN_MSR_DPS_CAL", "M3_ANCHOR_REVIN", "COFACTOR-44"),
        paired_delta(focus, "M11_REVIN_MSR_DPS_CAL", "M9_REVIN_MSR", "COFACTOR-44"),
    ], ignore_index=True)
    pairs.to_csv(OUT / "table_cofactor_44_module_pairwise.csv", index=False)
    write_markdown_table(pairs, OUT / "table_cofactor_44_module_pairwise.md")


def build_module_significance_summary() -> None:
    rows = [
        ["RevIN", "M3_ANCHOR_REVIN vs M1_ANCHOR", "COFACTOR-44", "all k=3/7/14/30", "win rate 95-100%; Wilcoxon p<0.0001", "core module"],
        ["MSR", "M9_REVIN_MSR vs M3_ANCHOR_REVIN", "COFACTOR-12", "all k=3/7/14/30", "win rate 92-100%; p<=0.0005", "core module"],
        ["MSR", "M9_REVIN_MSR vs M3_ANCHOR_REVIN", "BDG2-24", "all k=3/7/14/30", "win rate 79-88%; p<=0.0029", "core module"],
        ["DPS+Calibration", "M11_REVIN_MSR_DPS_CAL vs M3_ANCHOR_REVIN", "COFACTOR-12", "k=14/30 only", "significant for k=14/30; not significant for k=3/7", "conditional enhancement"],
        ["Ordinary similarity", "M2/M5 vs M1/M3", "COFACTOR-20", "mostly k=3/7/14/30", "not significant or harmful", "do not claim as core"],
        ["Gate", "M4/M6", "BDG2 pilot", "all k", "degraded performance", "remove"],
    ]
    df = pd.DataFrame(rows, columns=["module", "comparison", "dataset", "scope", "evidence", "decision"])
    df.to_csv(OUT / "table_module_decisions.csv", index=False)
    write_markdown_table(df, OUT / "table_module_decisions.md")


def _standardize_old_method(name: str) -> str:
    mapping = {
        "persistence": "Persistence",
        "random_forest_source_target": "RF-ST",
        "extra_trees_source_target": "ExtraTrees-ST",
        "hist_gbdt_source_target": "HistGBDT-ST",
        "PAR_HIST_GBDT_SW_CAL": "PARBoost",
        "seasonal_naive_24": "Seasonal naive",
    }
    return mapping.get(name, name)


def _wilcoxon_greater(delta: pd.Series) -> float | None:
    if wilcoxon is None or len(delta) < 6 or not (delta != 0).any():
        return None
    try:
        return float(wilcoxon(delta, alternative="greater").pvalue)
    except Exception:
        return None


def build_final_pairwise_robustness() -> None:
    old = pd.read_csv("results/parboost_evidence/table_all_validation_building_level_with_metadata.csv")
    old = old[old["dataset"].isin(["BDG2-24", "COFACTOR-44"])].copy()
    old["method"] = old["method"].map(_standardize_old_method)
    old = old[["dataset", "target", "method", "k", "MAE", "RMSE", "sMAPE", "sigma_err"]]

    bdg2 = pd.read_csv("results/pa_module_screen_bdg2_24_msr_dps/aggregate/table_pa_module_extensions.csv")
    bdg2["dataset"] = "BDG2-24"
    cofactor = pd.read_csv("results/pa_module_screen_cofactor_44_msr_dps/aggregate/table_pa_module_extensions.csv")
    cofactor["dataset"] = "COFACTOR-44"
    new = pd.concat([bdg2, cofactor], ignore_index=True)
    new = new[new["method"].isin(["M3_ANCHOR_REVIN", "M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL"])].copy()
    new = new[["dataset", "target", "method", "k", "MAE", "RMSE", "sMAPE", "sigma_err"]]

    all_rows = pd.concat([old, new], ignore_index=True)

    candidates = ["M9_REVIN_MSR", "M11_REVIN_MSR_DPS_CAL"]
    baselines = [
        "Persistence",
        "RF-ST",
        "ExtraTrees-ST",
        "HistGBDT-ST",
        "PARBoost",
        "M3_ANCHOR_REVIN",
    ]
    rows = []
    for dataset in ["BDG2-24", "COFACTOR-44"]:
        dfd = all_rows[all_rows["dataset"] == dataset]
        for candidate in candidates:
            for baseline in baselines:
                for k in sorted(dfd["k"].unique()):
                    a = dfd[(dfd["method"] == candidate) & (dfd["k"] == k)][["target", "MAE"]].rename(columns={"MAE": "candidate_MAE"})
                    b = dfd[(dfd["method"] == baseline) & (dfd["k"] == k)][["target", "MAE"]].rename(columns={"MAE": "baseline_MAE"})
                    m = a.merge(b, on="target")
                    if m.empty:
                        continue
                    delta = m["baseline_MAE"] - m["candidate_MAE"]
                    rows.append({
                        "dataset": dataset,
                        "candidate": candidate,
                        "baseline": baseline,
                        "k": int(k),
                        "n": int(len(m)),
                        "candidate_mean_MAE": float(m["candidate_MAE"].mean()),
                        "baseline_mean_MAE": float(m["baseline_MAE"].mean()),
                        "mean_improvement_MAE": float(delta.mean()),
                        "median_improvement_MAE": float(delta.median()),
                        "p10_improvement_MAE": float(np.percentile(delta, 10)),
                        "win_rate": float((delta > 0).mean()),
                        "negative_transfer_rate": float((delta < 0).mean()),
                        "wilcoxon_p_greater": _wilcoxon_greater(delta),
                    })
    out = pd.DataFrame(rows).sort_values(["dataset", "candidate", "baseline", "k"])
    out.to_csv(OUT / "table_final_pairwise_robustness.csv", index=False)
    write_markdown_table(out, OUT / "table_final_pairwise_robustness.md")

    rank_rows = []
    for dataset in ["BDG2-24", "COFACTOR-44"]:
        dfd = all_rows[all_rows["dataset"] == dataset]
        for k, g in dfd.groupby("k"):
            pivot = g.pivot_table(index="target", columns="method", values="MAE", aggfunc="mean")
            ranks = pivot.rank(axis=1, method="average", ascending=True)
            for method in sorted(pivot.columns):
                vals = pivot[method].dropna()
                if vals.empty:
                    continue
                rank_vals = ranks[method].dropna()
                rank_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "k": int(k),
                    "mean_MAE": float(vals.mean()),
                    "median_MAE": float(vals.median()),
                    "p90_MAE": float(np.percentile(vals, 90)),
                    "worst_MAE": float(vals.max()),
                    "mean_rank": float(rank_vals.mean()),
                    "top3_rate": float((rank_vals <= 3).mean()),
                    "n": int(len(vals)),
                })
    rank = pd.DataFrame(rank_rows).sort_values(["dataset", "k", "mean_MAE"])
    rank.to_csv(OUT / "table_final_leaderboard_ranked.csv", index=False)
    write_markdown_table(rank, OUT / "table_final_leaderboard_ranked.md")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_bdg2_candidate_main_table()
    if Path("results/pa_module_screen_cofactor_44_msr_dps/aggregate/table_pa_module_extensions_summary.csv").exists():
        build_cofactor_candidate_main_table()
        build_final_pairwise_robustness()
    summarize_module_with_active_split()
    build_module_significance_summary()
    print(f"Wrote final claim-evidence tables to {OUT}")


if __name__ == "__main__":
    main()
