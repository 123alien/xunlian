#!/usr/bin/env python
"""Audit whether extreme expanded results come from data, scripts, or model behavior."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import build_windows


BASE = Path("results/phase2_paper/expanded_fast")
DATA = Path("data/processed/bdg2_electricity_hourly_expanded.csv")
OUT = BASE / "diagnostics"
FEATURE_COLS = ["energy", "hour", "day_of_week", "is_weekend", "month"]


def split_target(df):
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def persistence_from_test(test_df):
    x, y = build_windows(test_df, FEATURE_COLS, "energy", 24, 1)
    pred = x[:, -1, 0]
    err = np.abs(y - pred)
    return y, pred, err


def load_npz_stats(path):
    if not path.exists():
        return {}
    z = np.load(path)
    y_true = z["y_true"]
    y_pred = z["y_pred"]
    err = np.abs(y_true - y_pred)
    return {
        "pred_exists": True,
        "pred_y_true_mean": float(np.mean(y_true)),
        "pred_y_true_std": float(np.std(y_true)),
        "pred_y_pred_mean": float(np.mean(y_pred)),
        "pred_y_pred_std": float(np.std(y_pred)),
        "pred_mae": float(np.mean(err)),
        "pred_bias": float(np.mean(y_pred - y_true)),
        "pred_corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8 else np.nan,
    }
    return {"pred_exists": False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    bres = pd.read_csv(BASE / "forecasting_paper/table_all_forecasting_methods_building_level.csv")
    focus_methods = [
        "persistence",
        "random_forest_source_target",
        "dlinear_M3_pretrain_ft",
        "dlinear_M3R_lambda0.3",
    ]

    rows = []
    for bid, g in data.groupby("building_id"):
        g = g.sort_values("timestamp")
        train, val, test = split_target(g)
        y_test, p_pred, p_err = persistence_from_test(test)
        train_energy = train["energy"].to_numpy(dtype=float)
        test_energy = test["energy"].to_numpy(dtype=float)
        diff = np.abs(np.diff(test_energy))
        row = {
            "target": bid,
            "n_total": len(g),
            "test_start": str(test["timestamp"].min()),
            "test_end": str(test["timestamp"].max()),
            "train_mean": float(np.mean(train_energy)),
            "train_std": float(np.std(train_energy)),
            "test_mean": float(np.mean(test_energy)),
            "test_std": float(np.std(test_energy)),
            "test_min": float(np.min(test_energy)),
            "test_max": float(np.max(test_energy)),
            "test_zero_ratio": float(np.mean(np.isclose(test_energy, 0.0))),
            "test_near_zero_ratio_1e_3": float(np.mean(np.abs(test_energy) < 1e-3)),
            "test_near_constant_ratio": float(np.mean(diff < 1e-6)) if len(diff) else np.nan,
            "test_diff_mean": float(np.mean(diff)) if len(diff) else np.nan,
            "test_diff_p95": float(np.percentile(diff, 95)) if len(diff) else np.nan,
            "manual_persistence_mae": float(np.mean(p_err)),
            "manual_persistence_rmse": float(np.sqrt(np.mean(p_err ** 2))),
            "train_test_mean_ratio": float(np.mean(test_energy) / np.mean(train_energy)) if np.mean(train_energy) else np.nan,
            "train_test_std_ratio": float(np.std(test_energy) / np.std(train_energy)) if np.std(train_energy) else np.nan,
        }
        for k in [3, 7, 14, 30]:
            sub = bres[(bres["target"] == bid) & (bres["k"] == k) & (bres["method"].isin(focus_methods))]
            for _, r in sub.iterrows():
                prefix = f"k{k}_{r['method']}"
                row[f"{prefix}_MAE"] = r["MAE"]
                row[f"{prefix}_sMAPE"] = r["sMAPE"]
                row[f"{prefix}_rank"] = r["rank_MAE"]

            # Check one representative DLinear-SRFT prediction file where available.
            pred_path = (
                BASE / "neural" / bid / "dlinear" / f"k{k}" / "seed42" / "M3R_lambda0.3" / "predictions.npz"
            )
            stats = load_npz_stats(pred_path)
            for key, value in stats.items():
                row[f"k{k}_srft_seed42_{key}"] = value
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "table_data_script_model_audit.csv", index=False)

    # Compare manual persistence to result table.
    ptab = bres[bres["method"] == "persistence"][["target", "k", "MAE", "RMSE", "sMAPE"]].copy()
    comp_rows = []
    for _, r in ptab.iterrows():
        a = audit[audit["target"] == r["target"]].iloc[0]
        comp_rows.append({
            "target": r["target"],
            "k": r["k"],
            "reported_persistence_mae": r["MAE"],
            "manual_persistence_mae": a["manual_persistence_mae"],
            "abs_diff": abs(r["MAE"] - a["manual_persistence_mae"]),
            "reported_sMAPE": r["sMAPE"],
        })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "table_persistence_script_check.csv", index=False)

    # Compact suspect tables.
    suspect_constant = audit.sort_values("manual_persistence_mae").head(12)
    suspect_constant.to_csv(OUT / "table_near_constant_test_buildings.csv", index=False)

    # Worst DLinear-SRFT k=3 bias/prediction checks.
    pred_cols = [
        "target", "test_mean", "test_std", "manual_persistence_mae",
        "k3_dlinear_M3R_lambda0.3_MAE", "k3_random_forest_source_target_MAE",
        "k3_srft_seed42_pred_y_true_mean", "k3_srft_seed42_pred_y_true_std",
        "k3_srft_seed42_pred_y_pred_mean", "k3_srft_seed42_pred_y_pred_std",
        "k3_srft_seed42_pred_bias", "k3_srft_seed42_pred_corr",
    ]
    audit[pred_cols].sort_values("k3_dlinear_M3R_lambda0.3_MAE", ascending=False).to_csv(
        OUT / "table_k3_srft_prediction_behavior.csv", index=False
    )

    print("[audit] saved to", OUT)
    print("\nPersistence script check max abs diff:", comp["abs_diff"].max())
    print("\nNear-constant / low-persistence buildings:")
    print(suspect_constant[[
        "target", "test_mean", "test_std", "test_zero_ratio", "manual_persistence_mae",
        "test_diff_mean", "test_diff_p95",
    ]].to_string(index=False))
    print("\nWorst k=3 SRFT prediction behavior:")
    print(audit[pred_cols].sort_values("k3_dlinear_M3R_lambda0.3_MAE", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
