#!/usr/bin/env python
"""Generate paired statistical tests for Phase 2 paper results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normal_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(x: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test using normal approximation.

    Returns (statistic W, p_value). Avoids scipy dependency.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    abs_x = np.abs(x)
    order = np.argsort(abs_x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_x[order[j + 1]] == abs_x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    w_pos = float(ranks[x > 0].sum())
    w_neg = float(ranks[x < 0].sum())
    w = min(w_pos, w_neg)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return w, float("nan")
    z = (w - mean) / np.sqrt(var)
    p = 2 * normal_cdf(z)
    return w, max(0.0, min(1.0, p))


def cliffs_delta(x: np.ndarray) -> float:
    """Cliff's delta against zero: positive means values tend to be > 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float((np.sum(x > 0) - np.sum(x < 0)) / len(x))


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_by_building_k_seed(few: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = few[few["method"].isin(["M2_target_only", "M3_pretrain_ft"])].copy()
    for key, g in subset.groupby(["target", "model", "k", "seed"]):
        m2 = g[g["method"] == "M2_target_only"]
        m3 = g[g["method"] == "M3_pretrain_ft"]
        if len(m2) and len(m3):
            m2 = m2.iloc[0]
            m3 = m3.iloc[0]
            row = {"target": key[0], "model": key[1], "k": key[2], "seed": key[3]}
            for metric in ["MAE", "RMSE", "sMAPE", "sigma_err", "sigma_score"]:
                row[f"delta_{metric}"] = m3[metric] - m2[metric]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--output-dir", default="results/phase2_paper/aggregate")
    args = ap.parse_args()

    agg = Path(args.aggregate_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    few = pd.read_csv(agg / "table_few_shot_transfer_all_models.csv")
    pair = paired_by_building_k_seed(few)
    pair.to_csv(out / "table_forecasting_paired_deltas.csv", index=False)

    rows = []
    for (model, k), g in pair.groupby(["model", "k"]):
        for metric in ["MAE", "RMSE", "sMAPE", "sigma_err", "sigma_score"]:
            vals = g[f"delta_{metric}"].to_numpy(float)
            w, p = wilcoxon_signed_rank(vals)
            ci_low, ci_high = bootstrap_ci_mean(vals)
            rows.append({
                "comparison": "M3_minus_M2",
                "model": model,
                "k": int(k),
                "metric": metric,
                "n_pairs": len(vals),
                "mean_delta": float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_W": w,
                "wilcoxon_p": p,
                "cliffs_delta_vs_zero": cliffs_delta(vals),
                "M3_win_rate": float(np.mean(vals < 0)),
                "interpretation": "M3 lower is better" if metric != "sigma_score" else "lower score variability is usually better",
            })
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "table_statistical_tests.csv", index=False)
    print(tests.to_string(index=False))
    print(f"[stats] saved to {out}")


if __name__ == "__main__":
    main()
