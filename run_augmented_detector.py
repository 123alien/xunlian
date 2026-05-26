#!/usr/bin/env python
"""Self-supervised calibration for cold-start energy anomaly detection.

The detector is calibrated on synthetic anomalies injected into the validation
period, then evaluated on a disjoint injected test period. Test labels are never
used for model selection or threshold selection.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from run_event_detector import (  # noqa: E402
    add_type_recalls,
    event_metrics_from_prediction,
    flatline_scores,
    merge_short_gaps,
    metrics_from_prediction,
    persist,
    rolling_slope_scores,
    safe_div,
    seasonal_mad_scores,
    val_test_hours,
)
from run_scoring_ablation import (  # noqa: E402
    cusum_scores,
    ewma_scores,
    prepare_arrays,
    quantile_threshold,
    zscore_scores,
)
from src.anomaly import rolling_mad_scores  # noqa: E402
from src.anomaly_injection import inject_anomalies  # noqa: E402
from src.config import Config  # noqa: E402
from src.phase2 import FEATURE_COLS, make_model  # noqa: E402
from src.train import make_dataloader, predict  # noqa: E402


CHANNELS = ["z", "mad", "seasonal", "ewma", "cusum", "flat", "slope", "hybrid"]
PERSISTENCE = [(1, 1), (2, 1), (3, 2), (4, 2), (6, 3), (8, 4)]
THRESHOLD_GRID = np.r_[np.linspace(0.5, 3.0, 26), np.linspace(3.25, 8.0, 20)]


def channel_scores(ref_err, sample_err, ref_y, sample_y, ref_hours, sample_hours, cfg):
    ref = {
        "z": zscore_scores(ref_err, ref_err),
        "mad": rolling_mad_scores(ref_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon),
        "seasonal": seasonal_mad_scores(ref_err, ref_hours, ref_err, ref_hours),
        "ewma": ewma_scores(ref_err, alpha=0.25),
        "cusum": cusum_scores(ref_err, ref_err, k=0.25),
        "flat": flatline_scores(ref_y, window=6),
        "slope": rolling_slope_scores(ref_y, window=12),
    }
    sample = {
        "z": zscore_scores(sample_err, ref_err),
        "mad": rolling_mad_scores(sample_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon),
        "seasonal": seasonal_mad_scores(sample_err, sample_hours, ref_err, ref_hours),
        "ewma": ewma_scores(sample_err, alpha=0.25),
        "cusum": cusum_scores(sample_err, ref_err, k=0.25),
        "flat": flatline_scores(sample_y, window=6),
        "slope": rolling_slope_scores(sample_y, window=12),
    }
    norm = {}
    for c, score in sample.items():
        q = 0.995 if c in {"z", "mad", "seasonal"} else 0.99
        norm[c] = safe_div(score, quantile_threshold(ref[c], q))
    norm["hybrid"] = np.max(np.vstack([norm[c] for c in ["z", "mad", "seasonal", "ewma", "cusum", "flat", "slope"]]), axis=0)
    return norm


def apply_rule(scores: np.ndarray, threshold: float, window: int, min_hits: int) -> np.ndarray:
    raw = scores > threshold
    if window > 1:
        raw = persist(raw.astype(int), window=window, min_hits=min_hits).astype(bool)
    return merge_short_gaps(raw.astype(int), max_gap=2)


def objective(labels: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> float:
    point_f1 = f1_score(labels, pred, zero_division=0)
    ev = event_metrics_from_prediction(labels, pred)
    far = pred[labels == 0].sum() / max((labels == 0).sum(), 1)
    try:
        auprc = average_precision_score(labels, scores)
    except ValueError:
        auprc = 0.0
    # Penalize noisy detectors hard; reviewers care about false alarms.
    return 0.45 * point_f1 + 0.45 * ev["EventF1"] + 0.20 * auprc - 0.35 * far


def choose_detector(labels: np.ndarray, score_map: dict[str, np.ndarray]) -> dict:
    best = None
    for channel in CHANNELS:
        scores = score_map[channel]
        finite = scores[np.isfinite(scores)]
        if len(finite) == 0:
            continue
        quantile_candidates = np.quantile(finite, [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99])
        thresholds = np.unique(np.r_[THRESHOLD_GRID, quantile_candidates])
        for threshold in thresholds:
            for window, min_hits in PERSISTENCE:
                pred = apply_rule(scores, float(threshold), window, min_hits)
                obj = objective(labels, scores, pred)
                if best is None or obj > best["objective"]:
                    best = {
                        "channel": channel,
                        "threshold": float(threshold),
                        "window": int(window),
                        "min_hits": int(min_hits),
                        "objective": float(obj),
                    }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    ap.add_argument("--phase2-dir", default="results/phase2_paper")
    ap.add_argument("--output-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--models", nargs="+", default=["lstm", "dlinear"])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--injection-rate", type=float, default=0.05)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    phase2 = Path(args.phase2_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    selected = []
    for target_id in sorted(data["building_id"].unique()):
        for model_type in args.models:
            for k in args.k:
                arrays = prepare_arrays(data, target_id, k, feature_cols, cfg)
                if arrays is None:
                    continue
                val_hours, test_hours = val_test_hours(arrays, cfg)
                for seed in args.seeds:
                    val_inj_y, val_labels, val_types = inject_anomalies(
                        arrays["y_val"], injection_rate=args.injection_rate,
                        seed=seed + 10000, return_types=True
                    )
                    for method_dir, method_name in [
                        ("M2", "M2_target_only"),
                        ("M3", "M3_pretrain_ft"),
                    ]:
                        ckpt = phase2 / target_id / model_type / f"k{k}" / f"seed{seed}" / method_dir / "model.pt"
                        if not ckpt.exists():
                            continue
                        model = make_model(model_type, len(feature_cols), cfg.data.window_size, cfg).to(device)
                        model.load_state_dict(torch.load(ckpt, map_location=device))
                        val_ldr = make_dataloader(arrays["X_val"], arrays["y_val_s"], cfg.train.batch_size, shuffle=False)
                        test_ldr = make_dataloader(arrays["X_test"], arrays["y_test_s"], cfg.train.batch_size, shuffle=False)
                        yv, pv = predict(model, val_ldr, device, y_scaler=arrays["y_scaler"])
                        yt, pt = predict(model, test_ldr, device, y_scaler=arrays["y_scaler"])
                        del model

                        clean_val_err = np.abs(yv - pv)
                        aug_val_err = np.abs(val_inj_y - pv)
                        test_err = np.abs(yt - pt)
                        val_scores = channel_scores(
                            clean_val_err, aug_val_err, arrays["y_val"], val_inj_y,
                            val_hours, val_hours, cfg
                        )
                        test_scores = channel_scores(
                            clean_val_err, test_err, arrays["y_val"], yt,
                            val_hours, test_hours, cfg
                        )
                        rule = choose_detector(val_labels, val_scores)
                        score = test_scores[rule["channel"]]
                        pred = apply_rule(score, rule["threshold"], rule["window"], rule["min_hits"])
                        row = {
                            "target": target_id,
                            "model": model_type,
                            "k": k,
                            "seed": seed,
                            "method": method_name,
                            "detector": "augmented_calibrated",
                            **rule,
                        }
                        row.update(metrics_from_prediction(arrays["labels"], score, pred))
                        add_type_recalls(row, arrays["labels"], pred, arrays["types"])
                        rows.append(row)
                        selected.append({
                            "target": target_id, "model": model_type, "k": k,
                            "seed": seed, "method": method_name, **rule,
                        })
                        print(f"[aug-detector] {target_id} {model_type} k{k} seed{seed} {method_name} {rule}")

    df = pd.DataFrame(rows)
    df.to_csv(output / "table_augmented_detector.csv", index=False)
    pd.DataFrame(selected).to_csv(output / "table_augmented_detector_selected_rules.csv", index=False)

    if not df.empty:
        metric_cols = [
            "Precision", "Recall", "F1", "AUROC", "AUPRC", "FAR",
            "EventPrecision", "EventRecall", "EventF1", "EventDelay",
        ]
        df.groupby(["model", "k", "method", "detector"])[metric_cols].mean().reset_index().to_csv(
            output / "table_augmented_detector_summary.csv", index=False
        )

        pairs = []
        for key, g in df.groupby(["target", "model", "k", "seed"]):
            m2 = g[g["method"] == "M2_target_only"]
            m3 = g[g["method"] == "M3_pretrain_ft"]
            if len(m2) and len(m3):
                m2 = m2.iloc[0]
                m3 = m3.iloc[0]
                pairs.append({
                    "target": key[0], "model": key[1], "k": key[2], "seed": key[3],
                    "delta_F1": m3["F1"] - m2["F1"],
                    "delta_AUPRC": m3["AUPRC"] - m2["AUPRC"],
                    "delta_FAR": m3["FAR"] - m2["FAR"],
                    "delta_EventF1": m3["EventF1"] - m2["EventF1"],
                    "M3_F1_win": m3["F1"] > m2["F1"],
                    "M3_AUPRC_win": m3["AUPRC"] > m2["AUPRC"],
                    "M3_FAR_win": m3["FAR"] < m2["FAR"],
                    "M3_EventF1_win": m3["EventF1"] > m2["EventF1"],
                })
        pair_df = pd.DataFrame(pairs)
        pair_df.to_csv(output / "table_augmented_detector_m2_m3_pairs.csv", index=False)
        if not pair_df.empty:
            claim = pair_df.groupby(["model", "k"]).agg(
                n=("target", "count"),
                F1_win_rate=("M3_F1_win", "mean"),
                AUPRC_win_rate=("M3_AUPRC_win", "mean"),
                FAR_win_rate=("M3_FAR_win", "mean"),
                EventF1_win_rate=("M3_EventF1_win", "mean"),
                mean_delta_F1=("delta_F1", "mean"),
                mean_delta_AUPRC=("delta_AUPRC", "mean"),
                mean_delta_FAR=("delta_FAR", "mean"),
                mean_delta_EventF1=("delta_EventF1", "mean"),
            ).reset_index()
            claim.to_csv(output / "table_augmented_detector_claim_summary.csv", index=False)
            print(claim.to_string(index=False))

    manifest = {
        "calibration": "detector channel, threshold, and persistence rule selected on validation-period synthetic anomalies only",
        "test_leakage": "no injected test labels are used for selection",
        "channels": CHANNELS,
        "persistence_grid": PERSISTENCE,
    }
    (output / "augmented_detector_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[aug-detector] saved rows={len(df)}")


if __name__ == "__main__":
    main()
