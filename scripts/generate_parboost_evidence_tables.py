#!/usr/bin/env python
"""Generate paper-facing evidence tables for PARBoost experiments."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DATASETS = {
    "bdg2_24": {
        "label": "BDG2-24",
        "building": "results/parboost_expanded_bdg2/validation_summary/table_validation_building_level.csv",
        "audit": "results/bdg2_full_eligible/table_eligibility_audit.csv",
    },
    "bdg2_120": {
        "label": "BDG2-120",
        "building": "results/bdg2_robustness_120/validation_summary/table_validation_building_level.csv",
        "audit": "results/bdg2_full_eligible/manifest_active_robustness_120.csv",
    },
    "cofactor_44": {
        "label": "COFACTOR-44",
        "building": "results/parboost_cofactor_full/table_validation_building_level.csv",
        "audit": None,
    },
}

METHODS = [
    "PAR_HIST_GBDT_SW_CAL",
    "persistence",
    "random_forest_source_target",
    "extra_trees_source_target",
    "hist_gbdt_source_target",
    "seasonal_naive_24",
]

BASELINES = [
    "persistence",
    "random_forest_source_target",
    "extra_trees_source_target",
    "hist_gbdt_source_target",
]


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 123, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def safe_wilcoxon(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 1e-12]
    if len(values) == 0:
        return np.nan
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def load_dataset(key: str) -> pd.DataFrame:
    cfg = DATASETS[key]
    df = pd.read_csv(cfg["building"])
    df = df[df["method"].isin(METHODS)].copy()
    df["dataset"] = cfg["label"]
    audit_path = cfg.get("audit")
    if audit_path and Path(audit_path).exists():
        audit = pd.read_csv(audit_path)
        keep = [
            c for c in [
                "building_id", "building_type", "test_std", "test_persistence_mae",
                "test_zero_ratio", "mean_energy", "std_energy", "zero_ratio",
            ]
            if c in audit.columns
        ]
        audit = audit[keep].drop_duplicates("building_id")
        df = df.merge(audit.rename(columns={"building_id": "target"}), on="target", how="left")
    return df


def tail_risk(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, k, method), g in df.groupby(["dataset", "k", "method"]):
        rel = g["relative_MAE_vs_persistence"].replace([np.inf, -np.inf], np.nan)
        rows.append({
            "dataset": dataset,
            "k": int(k),
            "method": method,
            "n_buildings": int(g["target"].nunique()),
            "mean_MAE": float(g["MAE"].mean()),
            "median_MAE": float(g["MAE"].median()),
            "p75_MAE": float(g["MAE"].quantile(0.75)),
            "p90_MAE": float(g["MAE"].quantile(0.90)),
            "p95_MAE": float(g["MAE"].quantile(0.95)),
            "worst_MAE": float(g["MAE"].max()),
            "mean_relative_MAE": float(rel.mean()),
            "median_relative_MAE": float(rel.median()),
            "relative_gt_1_rate": float((rel > 1).mean()),
            "mean_rank": float(g["rank"].mean()) if "rank" in g else np.nan,
            "top3_rate": float((g["rank"] <= 3).mean()) if "rank" in g else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "k", "mean_rank", "mean_MAE"])


def paired_tests(df: pd.DataFrame, candidate: str = "PAR_HIST_GBDT_SW_CAL") -> pd.DataFrame:
    rows = []
    for dataset, dset in df.groupby("dataset"):
        piv = dset.pivot_table(index=["target", "k"], columns="method", values="MAE", aggfunc="mean")
        for baseline in BASELINES:
            if candidate not in piv.columns or baseline not in piv.columns:
                continue
            tmp = (piv[candidate] - piv[baseline]).reset_index(name="delta").dropna()
            ratio = (piv[candidate] / piv[baseline]).reset_index(name="ratio").replace([np.inf, -np.inf], np.nan)
            tmp = tmp.merge(ratio, on=["target", "k"], how="left")
            for k, g in tmp.groupby("k"):
                delta = g["delta"].to_numpy(dtype=float)
                ratio_vals = g["ratio"].to_numpy(dtype=float)
                lo, hi = bootstrap_ci(delta)
                rows.append({
                    "dataset": dataset,
                    "candidate": candidate,
                    "baseline": baseline,
                    "k": int(k),
                    "n": int(len(g)),
                    "mean_delta": float(np.mean(delta)),
                    "mean_delta_ci95_low": lo,
                    "mean_delta_ci95_high": hi,
                    "median_delta": float(np.median(delta)),
                    "win_rate": float(np.mean(delta < 0)),
                    "loss_rate": float(np.mean(delta > 0)),
                    "p90_delta": float(np.quantile(delta, 0.90)),
                    "worst_delta": float(np.max(delta)),
                    "mean_ratio": float(np.nanmean(ratio_vals)),
                    "median_ratio": float(np.nanmedian(ratio_vals)),
                    "wilcoxon_p": safe_wilcoxon(delta),
                })
    return pd.DataFrame(rows).sort_values(["dataset", "baseline", "k"])


def stratify_by_persistence(df: pd.DataFrame, candidate: str = "PAR_HIST_GBDT_SW_CAL") -> pd.DataFrame:
    rows = []
    for dataset, dset in df.groupby("dataset"):
        piv = dset.pivot_table(index=["target", "k"], columns="method", values="MAE", aggfunc="mean")
        if candidate not in piv.columns or "persistence" not in piv.columns:
            continue
        tmp = pd.DataFrame({
            "par_mae": piv[candidate],
            "persistence_mae": piv["persistence"],
        }).reset_index().dropna()
        for k, g in tmp.groupby("k"):
            # Quantile bins can collapse when persistence is identical/tied; use rank fallback.
            ranks = g["persistence_mae"].rank(method="first")
            try:
                g = g.assign(persistence_bin=pd.qcut(ranks, 3, labels=["low", "mid", "high"]))
            except ValueError:
                g = g.assign(persistence_bin="all")
            for b, h in g.groupby("persistence_bin", observed=False):
                delta = h["par_mae"] - h["persistence_mae"]
                rows.append({
                    "dataset": dataset,
                    "k": int(k),
                    "stratum": str(b),
                    "n": int(len(h)),
                    "mean_persistence_MAE": float(h["persistence_mae"].mean()),
                    "mean_PARBoost_MAE": float(h["par_mae"].mean()),
                    "mean_delta_vs_persistence": float(delta.mean()),
                    "median_delta_vs_persistence": float(delta.median()),
                    "win_rate_vs_persistence": float((delta < 0).mean()),
                })
    return pd.DataFrame(rows).sort_values(["dataset", "k", "stratum"])


def stratify_by_audit(df: pd.DataFrame, candidate: str = "PAR_HIST_GBDT_SW_CAL") -> pd.DataFrame:
    rows = []
    available = [c for c in ["test_std", "test_persistence_mae", "test_zero_ratio"] if c in df.columns]
    if not available:
        return pd.DataFrame()
    for dataset, dset in df.groupby("dataset"):
        piv = dset.pivot_table(index=["target", "k"], columns="method", values="MAE", aggfunc="mean").reset_index()
        meta_cols = ["target", *available]
        meta = dset[meta_cols].dropna(subset=available, how="all").drop_duplicates("target")
        piv = piv.merge(meta, on="target", how="left")
        if candidate not in piv.columns or "persistence" not in piv.columns:
            continue
        for feature in available:
            valid = piv.dropna(subset=[feature]).copy()
            if valid.empty:
                continue
            ranks = valid[feature].rank(method="first")
            try:
                valid["bin"] = pd.qcut(ranks, 3, labels=["low", "mid", "high"])
            except ValueError:
                valid["bin"] = "all"
            for (k, b), g in valid.groupby(["k", "bin"], observed=False):
                delta_p = g[candidate] - g["persistence"]
                rows.append({
                    "dataset": dataset,
                    "feature": feature,
                    "k": int(k),
                    "stratum": str(b),
                    "n": int(len(g)),
                    "feature_mean": float(g[feature].mean()),
                    "PARBoost_mean_MAE": float(g[candidate].mean()),
                    "persistence_mean_MAE": float(g["persistence"].mean()),
                    "mean_delta_vs_persistence": float(delta_p.mean()),
                    "win_rate_vs_persistence": float((delta_p < 0).mean()),
                })
    return pd.DataFrame(rows).sort_values(["dataset", "feature", "k", "stratum"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/parboost_evidence")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = [load_dataset(k) for k in DATASETS]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "table_all_validation_building_level_with_metadata.csv", index=False)

    tables = {
        "table_tail_risk_by_dataset_method.csv": tail_risk(df),
        "table_paired_tests_parboost.csv": paired_tests(df),
        "table_stratified_by_persistence_strength.csv": stratify_by_persistence(df),
        "table_stratified_by_dataset_audit_features.csv": stratify_by_audit(df),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
        print(f"\n{name}")
        print(table.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
