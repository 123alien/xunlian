"""Phase 1: Core claim validation experiment.

Tests whether Pretrain + Fine-tune (M3) produces more stable prediction
errors and anomaly scores than Target-only few-shot (M2).

Setup:
- 3–5 buildings selected from BDG2
- LSTM only
- k = 7 days few-shot data
- seed = 42
- M2: train LSTM from scratch on 7 days of target building
- M3: pretrain LSTM on all source buildings, fine-tune on 7 days of target

Output:
- results/phase1/<building>/M2/ and M3/ with config, metrics, predictions
- results/phase1/core_claim_check.csv
- results/phase1/core_claim_check.md
"""
import sys
from pathlib import Path
# Ensure project root is on path before any src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.models import LSTMForecaster
from src.train import (
    build_windows, fit_scaler, apply_scaler, make_dataloader,
    train_model, pretrain, fine_tune, predict, save_run,
)
from src.anomaly import rolling_mad_scores
from src.metrics import compute_all


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_COLS = ["energy", "hour", "day_of_week", "is_weekend", "month"]
K_DAYS = 7
K_HOURS = K_DAYS * 24  # 168


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_phase1(config_path: str = "configs/default.yaml",
               data_path: str = "data/processed/bdg2_electricity_hourly.csv",
               output_dir: str = "results/phase1",
               n_buildings: int = 5):
    """Run the Phase 1 core claim validation experiment."""

    # Load config
    cfg = Config.from_yaml(config_path)
    cfg.seed = 42
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase1] Device: {device}")

    # Load data
    data = pd.read_csv(data_path, parse_dates=["timestamp"])
    building_ids = sorted(data["building_id"].unique())
    if len(building_ids) > n_buildings:
        building_ids = building_ids[:n_buildings]
    print(f"[Phase1] Buildings: {building_ids}")

    n_features = len(FEATURE_COLS)

    # Feature column check — fall back to what's available
    available_cols = [c for c in FEATURE_COLS if c in data.columns]
    if len(available_cols) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_cols)
        print(f"[Phase1] WARNING: Missing feature columns: {missing}")
        print(f"[Phase1] Using available features: {available_cols}")
    feature_cols = available_cols

    all_results = []

    # For each building as target
    for tgt_idx, target_id in enumerate(building_ids):
        print(f"\n{'='*60}")
        print(f"[Phase1] Target building {tgt_idx+1}/{len(building_ids)}: {target_id}")
        print(f"{'='*60}")

        source_ids = [b for b in building_ids if b != target_id]
        if len(source_ids) == 0:
            print("[Phase1] WARNING: Only 1 building -- skipping (need >=2 for transfer).")
            continue

        # ---- Prepare target building data ----
        tgt_data = data[data["building_id"] == target_id].sort_values("timestamp")

        # Chronological split
        n = len(tgt_data)
        train_end = int(n * 0.6)
        val_end = int(n * (0.6 + 0.2))
        tgt_train_full = tgt_data.iloc[:train_end]
        tgt_val = tgt_data.iloc[train_end:val_end]
        tgt_test = tgt_data.iloc[val_end:]

        # Take first K_HOURS rows of training portion as few-shot data
        tgt_train_few = tgt_train_full.iloc[:K_HOURS]

        print(f"  Target: {n} total rows, {len(tgt_train_few)} few-shot, "
              f"{len(tgt_val)} val, {len(tgt_test)} test")

        if len(tgt_train_few) < cfg.data.window_size + 1:
            print(f"  SKIP: insufficient few-shot data ({len(tgt_train_few)} rows)")
            continue

        # ---- Prepare source building data (all source buildings merged) ----
        source_loaders = []
        for src_id in source_ids:
            src_data = data[data["building_id"] == src_id].sort_values("timestamp")
            n_src = len(src_data)
            src_train = src_data.iloc[:int(n_src * 0.6)]
            src_val = src_data.iloc[int(n_src * 0.6):int(n_src * 0.8)]

            X_src_train, y_src_train = build_windows(
                src_train, feature_cols, "energy",
                cfg.data.window_size, cfg.data.horizon
            )
            X_src_val, y_src_val = build_windows(
                src_val, feature_cols, "energy",
                cfg.data.window_size, cfg.data.horizon
            )

            if len(X_src_train) == 0:
                continue

            mean_s, std_s = fit_scaler(X_src_train)
            X_src_train_s = apply_scaler(X_src_train, mean_s, std_s)
            X_src_val_s = apply_scaler(X_src_val, mean_s, std_s)

            src_train_ldr = make_dataloader(X_src_train_s, y_src_train,
                                            cfg.train.batch_size, shuffle=True)
            src_val_ldr = make_dataloader(X_src_val_s, y_src_val,
                                          cfg.train.batch_size, shuffle=False)
            source_loaders.append((src_train_ldr, src_val_ldr))

        if len(source_loaders) == 0:
            print("  SKIP: no valid source buildings.")
            continue

        print(f"  Source buildings: {len(source_loaders)}")

        # ---- Prepare target test data (shared for M2 and M3) ----
        X_test, y_test = build_windows(tgt_test, feature_cols, "energy",
                                       cfg.data.window_size, cfg.data.horizon)
        if len(X_test) == 0:
            print("  SKIP: insufficient test data.")
            continue

        # ---- Prepare few-shot train/val loaders (shared) ----
        X_few, y_few = build_windows(tgt_train_few, feature_cols, "energy",
                                     cfg.data.window_size, cfg.data.horizon)
        X_val, y_val = build_windows(tgt_val, feature_cols, "energy",
                                     cfg.data.window_size, cfg.data.horizon)

        few_mean, few_std = fit_scaler(X_few)
        X_few_s = apply_scaler(X_few, few_mean, few_std)
        X_val_s = apply_scaler(X_val, few_mean, few_std)
        X_test_s = apply_scaler(X_test, few_mean, few_std)

        # Use smaller batch size for few-shot data
        few_bs = min(cfg.train.batch_size, len(X_few_s) // 4)
        few_bs = max(few_bs, 4)  # minimum batch size of 4

        few_train_ldr = make_dataloader(X_few_s, y_few, few_bs, shuffle=True)
        few_val_ldr = make_dataloader(X_val_s, y_val, few_bs, shuffle=False)
        test_ldr = make_dataloader(X_test_s, y_test, cfg.train.batch_size, shuffle=False)

        # ================================================================
        # M2: Target-only few-shot
        # ================================================================
        print(f"\n  [M2] Target-only few-shot training ({K_DAYS} days, batch_size={few_bs})...")
        set_seed(cfg.seed)

        model_m2 = LSTMForecaster(
            input_dim=len(feature_cols),
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

        history_m2 = train_model(model_m2, few_train_ldr, few_val_ldr, cfg, device)

        y_true_m2, y_pred_m2 = predict(model_m2, test_ldr, device)
        errors_m2 = np.abs(y_true_m2 - y_pred_m2)
        scores_m2 = rolling_mad_scores(errors_m2, window=cfg.anomaly.mad_window,
                                       epsilon=cfg.anomaly.epsilon)
        metrics_m2 = compute_all(y_true_m2, y_pred_m2, scores_m2)
        metrics_m2["method"] = "M2_target_only"

        save_run(f"{output_dir}/{target_id}/M2", cfg, metrics_m2,
                 predictions=(y_true_m2, y_pred_m2),
                 anomaly_scores=scores_m2,
                 history=history_m2)

        print(f"    MAE={metrics_m2['MAE']:.4f}, RMSE={metrics_m2['RMSE']:.4f}, "
              f"sigma_err={metrics_m2['sigma_err']:.4f}, sigma_score={metrics_m2['sigma_score']:.4f}")

        # ================================================================
        # M3: Pretrain + Fine-tune
        # ================================================================
        print(f"  [M3] Pretraining on {len(source_loaders)} source buildings...")
        set_seed(cfg.seed)

        model_m3 = LSTMForecaster(
            input_dim=len(feature_cols),
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

        history_pretrain = pretrain(model_m3, source_loaders, cfg, device)
        print(f"  [M3] Fine-tuning on {K_DAYS} days of target...")
        history_ft = fine_tune(model_m3, few_train_ldr, few_val_ldr, cfg, device)

        y_true_m3, y_pred_m3 = predict(model_m3, test_ldr, device)
        errors_m3 = np.abs(y_true_m3 - y_pred_m3)
        scores_m3 = rolling_mad_scores(errors_m3, window=cfg.anomaly.mad_window,
                                       epsilon=cfg.anomaly.epsilon)
        metrics_m3 = compute_all(y_true_m3, y_pred_m3, scores_m3)
        metrics_m3["method"] = "M3_pretrain_ft"

        save_run(f"{output_dir}/{target_id}/M3", cfg, metrics_m3,
                 predictions=(y_true_m3, y_pred_m3),
                 anomaly_scores=scores_m3,
                 history={"pretrain": history_pretrain, "finetune": history_ft})

        print(f"    MAE={metrics_m3['MAE']:.4f}, RMSE={metrics_m3['RMSE']:.4f}, "
              f"sigma_err={metrics_m3['sigma_err']:.4f}, sigma_score={metrics_m3['sigma_score']:.4f}")

        # ---- Compare ----
        delta = {
            "target_building": target_id,
            "MAE_M2": metrics_m2["MAE"], "MAE_M3": metrics_m3["MAE"],
            "MAE_delta": metrics_m3["MAE"] - metrics_m2["MAE"],
            "RMSE_M2": metrics_m2["RMSE"], "RMSE_M3": metrics_m3["RMSE"],
            "RMSE_delta": metrics_m3["RMSE"] - metrics_m2["RMSE"],
            "sigma_err_M2": metrics_m2["sigma_err"],
            "sigma_err_M3": metrics_m3["sigma_err"],
            "sigma_err_delta": metrics_m3["sigma_err"] - metrics_m2["sigma_err"],
            "sigma_score_M2": metrics_m2["sigma_score"],
            "sigma_score_M3": metrics_m3["sigma_score"],
            "sigma_score_delta": metrics_m3["sigma_score"] - metrics_m2["sigma_score"],
            "sigma_err_win": metrics_m3["sigma_err"] < metrics_m2["sigma_err"],
            "sigma_score_win": metrics_m3["sigma_score"] < metrics_m2["sigma_score"],
        }
        all_results.append(delta)

        print(f"  d_sigma_err = {delta['sigma_err_delta']:+.4f} "
              f"({'M3 wins' if delta['sigma_err_win'] else 'M2 wins'})")
        print(f"  d_sigma_score = {delta['sigma_score_delta']:+.4f} "
              f"({'M3 wins' if delta['sigma_score_win'] else 'M2 wins'})")

    # ================================================================
    # Aggregate results
    # ================================================================
    if not all_results:
        print("\n[Phase1] ERROR: No valid results generated.")
        return

    results_df = pd.DataFrame(all_results)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "core_claim_check.csv", index=False)
    print(f"\n[Phase1] Results saved to {out_dir / 'core_claim_check.csv'}")

    # Summary
    n_win_err = results_df["sigma_err_win"].sum()
    n_win_score = results_df["sigma_score_win"].sum()
    n_total = len(results_df)
    mean_delta_err = results_df["sigma_err_delta"].mean()
    mean_delta_score = results_df["sigma_score_delta"].mean()

    print(f"\n[Phase1] SUMMARY:")
    print(f"  sigma_err:  M3 better in {n_win_err}/{n_total} buildings, mean delta = {mean_delta_err:+.4f}")
    print(f"  sigma_score: M3 better in {n_win_score}/{n_total} buildings, mean delta = {mean_delta_score:+.4f}")

    claim_supported = (n_win_err >= n_total / 2) and (n_win_score >= n_total / 2)
    verdict = (
        "SUPPORTED: M3 (pretrain+FT) produces more stable predictions "
        "than M2 (target-only) in the majority of buildings."
        if claim_supported else
        "NOT SUPPORTED: M3 does NOT consistently outperform M2 on stability metrics."
    )

    # Generate core_claim_check.md
    md_lines = [
        "# Phase 1: Core Claim Validation Report",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Seed**: {cfg.seed}",
        f"**k**: {K_DAYS} days ({K_HOURS} hours)",
        f"**Buildings**: {len(building_ids)} ({', '.join(building_ids)})",
        f"**Model**: LSTM (hidden={cfg.model.hidden_dim}, layers={cfg.model.num_layers})",
        "",
        "## Core Claim",
        "",
        "> Pretrain + fine-tune (M3) produces **lower σ_err and σ_score** ",
        "> than target-only few-shot (M2), indicating more stable prediction ",
        "> errors and more reliable anomaly scoring.",
        "",
        "## Results",
        "",
        f"| Target Building | σ_err M2 | σ_err M3 | Δ σ_err | σ_score M2 | σ_score M3 | Δ σ_score | σ_err Win? | σ_score Win? |",
        f"|{'—'*16}|{'—'*10}|{'—'*10}|{'—'*9}|{'—'*12}|{'—'*12}|{'—'*11}|{'—'*11}|{'—'*13}|",
    ]
    for _, row in results_df.iterrows():
        md_lines.append(
            f"| {row['target_building']:<16} | {row['sigma_err_M2']:.4f} | "
            f"{row['sigma_err_M3']:.4f} | {row['sigma_err_delta']:+.4f} | "
            f"{row['sigma_score_M2']:.4f} | {row['sigma_score_M3']:.4f} | "
            f"{row['sigma_score_delta']:+.4f} | "
            f"{'Win' if row['sigma_err_win'] else 'Lose'} | "
            f"{'Win' if row['sigma_score_win'] else 'Lose'} |"
        )

    md_lines += [
        "",
        "## Summary",
        "",
        f"- **σ_err**: M3 wins in {n_win_err}/{n_total} buildings (mean Δ = {mean_delta_err:+.4f})",
        f"- **σ_score**: M3 wins in {n_win_score}/{n_total} buildings (mean Δ = {mean_delta_score:+.4f})",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "### Interpretation",
        "",
    ]

    if claim_supported:
        md_lines += [
            "The core claim is supported. Transfer learning (pretrain + fine-tune) ",
            "improves prediction stability compared to training on limited target data alone. ",
            "This provides evidence that the TL-TFAD framework is viable, and the project ",
            "should proceed to Phase 2 (full experiment matrix).",
        ]
    else:
        md_lines += [
            "The core claim is NOT supported by these results. Possible reasons:",
            "",
            "1. 7 days of data may be insufficient even for fine-tuning.",
            "2. The selected buildings may have fundamentally different load patterns (negative transfer).",
            "3. LSTM architecture or hyperparameters may need tuning.",
            "4. The pretraining procedure may not be learning transferable patterns.",
            "",
            "**Recommended next steps**:",
            "- Try k=14 days instead of 7.",
            "- Try different pretraining hyperparameters (more epochs, lower LR).",
            "- Check whether the source buildings have enough data for meaningful pretraining.",
            "- Try GRU or DLinear as alternative backbones.",
        ]

    md_lines += [
        "",
        "## Next Steps",
        "",
        "If Phase 1 passes: proceed to `src/phase2.py` — full experiment matrix with ",
        "k ∈ {0, 1, 3, 7, 14}, DLinear baseline, 5+ buildings, anomaly injection, ",
        "and unsupervised AD baselines.",
    ]

    with open(out_dir / "core_claim_check.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[Phase1] Report saved to {out_dir / 'core_claim_check.md'}")
    print(f"[Phase1] VERDICT: {verdict}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: Core claim validation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    parser.add_argument("--output", default="results/phase1")
    parser.add_argument("--n-buildings", type=int, default=5)
    parser.add_argument("--k-days", type=int, default=7)
    args = parser.parse_args()

    K_DAYS = args.k_days
    K_HOURS = K_DAYS * 24

    run_phase1(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        n_buildings=args.n_buildings,
    )
