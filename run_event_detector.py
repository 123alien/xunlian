#!/usr/bin/env python
"""Evaluate calibrated event-level anomaly detectors on Phase 2 checkpoints.

This script is intentionally train-free: it reuses the saved M2/M3 forecasting
checkpoints, calibrates detector thresholds on clean validation residuals, and
evaluates on injected test anomalies. The goal is to test whether the anomaly
detection layer, not just the forecaster, can support the paper's main claim.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))

from run_scoring_ablation import (  # noqa: E402
    contiguous_events,
    cusum_scores,
    ewma_scores,
    flatline_scores,
    prepare_arrays,
    quantile_threshold,
    zscore_scores,
)
from src.anomaly import rolling_mad_scores  # noqa: E402
from src.config import Config  # noqa: E402
from src.phase2 import FEATURE_COLS, make_model  # noqa: E402
from src.train import make_dataloader, predict  # noqa: E402


DETECTORS = [
    "point_residual_q995",
    "seasonal_mad_q99",
    "conformal_abs_q99",
    "event_ewma_cusum",
    "event_hybrid_conformal",
]


def safe_div(a: np.ndarray, b: float) -> np.ndarray:
    if not np.isfinite(b) or abs(b) < 1e-8:
        b = 1.0
    return a / b


def seasonal_mad_scores(errors: np.ndarray, hours: np.ndarray,
                        ref_errors: np.ndarray, ref_hours: np.ndarray) -> np.ndarray:
    """Hour-of-day robust residual score."""
    out = np.zeros_like(errors, dtype=float)
    global_med = float(np.median(ref_errors))
    global_mad = float(np.median(np.abs(ref_errors - global_med)))
    global_mad = max(global_mad, 1e-6)
    for h in range(24):
        ref_mask = ref_hours == h
        test_mask = hours == h
        if not np.any(test_mask):
            continue
        ref = ref_errors[ref_mask]
        if len(ref) < 5:
            med, mad = global_med, global_mad
        else:
            med = float(np.median(ref))
            mad = float(np.median(np.abs(ref - med)))
            mad = max(mad, global_mad * 0.05, 1e-6)
        out[test_mask] = np.maximum(0.0, errors[test_mask] - med) / mad
    return np.clip(out, 0.0, 100.0)


def rolling_slope_scores(y: np.ndarray, window: int = 12) -> np.ndarray:
    """Simple local trend magnitude score for slow drift events."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=float)
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x ** 2))
    scale = max(float(np.std(np.diff(y, prepend=y[0]))), 1e-6)
    for i in range(len(y)):
        start = max(0, i - window + 1)
        seg = y[start:i + 1]
        if len(seg) < 3:
            continue
        xx = np.arange(len(seg), dtype=float)
        xx = xx - xx.mean()
        slope = float(np.sum(xx * (seg - seg.mean())) / max(np.sum(xx ** 2), 1e-8))
        out[i] = abs(slope) / scale
    return np.clip(out, 0.0, 100.0)


def persist(binary: np.ndarray, window: int, min_hits: int) -> np.ndarray:
    """Require at least min_hits alarms within a trailing window."""
    binary = binary.astype(int)
    out = np.zeros_like(binary)
    csum = np.cumsum(np.r_[0, binary])
    for i in range(len(binary)):
        start = max(0, i - window + 1)
        hits = csum[i + 1] - csum[start]
        out[i] = int(hits >= min_hits)
    return out


def merge_short_gaps(binary: np.ndarray, max_gap: int = 2) -> np.ndarray:
    out = binary.astype(int).copy()
    events = contiguous_events(out)
    for left, right in zip(events, events[1:]):
        gap_start = left[1] + 1
        gap_end = right[0] - 1
        if 0 <= gap_end - gap_start + 1 <= max_gap:
            out[gap_start:gap_end + 1] = 1
    return out


def build_scores(val_err: np.ndarray, test_err: np.ndarray,
                 val_y: np.ndarray, test_y: np.ndarray,
                 val_hours: np.ndarray, test_hours: np.ndarray,
                 cfg: Config) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    val_parts = {
        "z": zscore_scores(val_err, val_err),
        "mad": rolling_mad_scores(val_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon),
        "ewma": ewma_scores(val_err, alpha=0.25),
        "cusum": cusum_scores(val_err, val_err, k=0.25),
        "seasonal": seasonal_mad_scores(val_err, val_hours, val_err, val_hours),
        "flat": flatline_scores(val_y, window=6),
        "slope": rolling_slope_scores(val_y, window=12),
    }
    test_parts = {
        "z": zscore_scores(test_err, val_err),
        "mad": rolling_mad_scores(test_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon),
        "ewma": ewma_scores(test_err, alpha=0.25),
        "cusum": cusum_scores(test_err, val_err, k=0.25),
        "seasonal": seasonal_mad_scores(test_err, test_hours, val_err, val_hours),
        "flat": flatline_scores(test_y, window=6),
        "slope": rolling_slope_scores(test_y, window=12),
    }
    return val_parts, test_parts


def detector_output(name: str, val_parts: dict[str, np.ndarray],
                    test_parts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, str]:
    """Return continuous score, binary alarm, and calibration description."""
    if name == "point_residual_q995":
        val_score = val_parts["z"]
        test_score = test_parts["z"]
        thr = quantile_threshold(val_score, 0.995)
        pred = test_score > thr
        return test_score, pred.astype(int), f"z>val_q995({thr:.4g})"

    if name == "seasonal_mad_q99":
        val_score = val_parts["seasonal"]
        test_score = test_parts["seasonal"]
        thr = quantile_threshold(val_score, 0.99)
        pred = merge_short_gaps(persist(test_score > thr, window=2, min_hits=1))
        return test_score, pred.astype(int), f"seasonal_mad>val_q99({thr:.4g})"

    if name == "conformal_abs_q99":
        val_score = val_parts["z"]
        test_score = test_parts["z"]
        thr = quantile_threshold(val_score, 0.99)
        pred = merge_short_gaps(persist(test_score > thr, window=3, min_hits=2))
        return test_score, pred.astype(int), f"abs_residual conformal q99({thr:.4g}) + 2/3"

    if name == "event_ewma_cusum":
        ewma_thr = quantile_threshold(val_parts["ewma"], 0.99)
        cusum_thr = quantile_threshold(val_parts["cusum"], 0.99)
        ewma_alarm = persist(test_parts["ewma"] > ewma_thr, window=3, min_hits=2)
        cusum_alarm = persist(test_parts["cusum"] > cusum_thr, window=6, min_hits=3)
        score = np.maximum(
            safe_div(test_parts["ewma"], ewma_thr),
            safe_div(test_parts["cusum"], cusum_thr),
        )
        pred = merge_short_gaps((ewma_alarm | cusum_alarm).astype(int))
        return score, pred.astype(int), "EWMA q99 2/3 OR CUSUM q99 3/6"

    if name == "event_hybrid_conformal":
        channels = ["z", "mad", "seasonal", "ewma", "cusum", "flat", "slope"]
        val_norm = {}
        test_norm = {}
        for channel in channels:
            q = 0.995 if channel in {"z", "mad", "seasonal"} else 0.99
            scale = quantile_threshold(val_parts[channel], q)
            val_norm[channel] = safe_div(val_parts[channel], scale)
            test_norm[channel] = safe_div(test_parts[channel], scale)

        point = (
            (test_norm["z"] > 1.0)
            | (test_norm["mad"] > 1.0)
            | (test_norm["seasonal"] > 1.0)
        )
        sustained_raw = (
            (test_norm["ewma"] > 1.0)
            | (test_norm["cusum"] > 1.0)
            | (test_norm["slope"] > 1.0)
        )
        flat_raw = test_norm["flat"] > 1.0
        sustained = persist(sustained_raw.astype(int), window=4, min_hits=2)
        flat = persist(flat_raw.astype(int), window=6, min_hits=3)
        score = np.max(np.vstack([test_norm[c] for c in channels]), axis=0)
        pred = merge_short_gaps((point | sustained | flat).astype(int), max_gap=2)
        return score, pred.astype(int), "max calibrated channels; point OR 2/4 sustained OR 3/6 flat"

    raise ValueError(f"Unknown detector: {name}")


def metrics_from_prediction(labels: np.ndarray, scores: np.ndarray,
                            pred: np.ndarray) -> dict:
    labels = labels.astype(int)
    pred = pred.astype(int)
    out = {
        "Precision": float(precision_score(labels, pred, zero_division=0)),
        "Recall": float(recall_score(labels, pred, zero_division=0)),
        "F1": float(f1_score(labels, pred, zero_division=0)),
        "FAR": float(pred[labels == 0].sum() / max((labels == 0).sum(), 1)),
    }
    try:
        out["AUROC"] = float(roc_auc_score(labels, scores))
        out["AUPRC"] = float(average_precision_score(labels, scores))
    except ValueError:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
    out.update(event_metrics_from_prediction(labels, pred))
    return out


def event_metrics_from_prediction(labels: np.ndarray, pred: np.ndarray) -> dict:
    true_events = contiguous_events(labels)
    pred_events = contiguous_events(pred)
    detected = 0
    delays = []
    for te in true_events:
        hits = [pe for pe in pred_events if max(te[0], pe[0]) <= min(te[1], pe[1])]
        if hits:
            detected += 1
            first_hit = min(max(pe[0], te[0]) for pe in hits)
            delays.append(max(0, first_hit - te[0]))
    matched_pred = sum(
        int(any(max(pe[0], te[0]) <= min(pe[1], te[1]) for te in true_events))
        for pe in pred_events
    )
    ep = matched_pred / max(len(pred_events), 1)
    er = detected / max(len(true_events), 1)
    ef1 = 2 * ep * er / (ep + er) if ep + er > 0 else 0.0
    return {
        "EventPrecision": float(ep),
        "EventRecall": float(er),
        "EventF1": float(ef1),
        "EventDelay": float(np.mean(delays)) if delays else float("nan"),
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
    }


def add_type_recalls(row: dict, labels: np.ndarray, pred: np.ndarray, types: np.ndarray):
    for anomaly_type in sorted(set(types) - {"normal"}):
        mask = types == anomaly_type
        if mask.sum() == 0:
            continue
        row[f"Recall_{anomaly_type}"] = float(
            recall_score(labels[mask], pred[mask], zero_division=0)
        )


def val_test_hours(arrays: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    val_hours = arrays["X_val"][:, -1, FEATURE_COLS.index("hour")]
    test_hours = arrays["X_test"][:, -1, FEATURE_COLS.index("hour")]
    # X arrays are scaled, so fall back to sequential hour bins when unavailable.
    if np.nanmax(np.abs(val_hours)) <= 5:
        val_hours = np.arange(len(arrays["y_val"])) % 24
        test_hours = np.arange(len(arrays["y_test"])) % 24
    return val_hours.astype(int) % 24, test_hours.astype(int) % 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    ap.add_argument("--phase2-dir", default="results/phase2_paper")
    ap.add_argument("--output-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--models", nargs="+", default=["lstm", "dlinear"])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    phase2 = Path(args.phase2_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    manifests = []
    for target_id in sorted(data["building_id"].unique()):
        for model_type in args.models:
            for k in args.k:
                arrays = prepare_arrays(data, target_id, k, feature_cols, cfg)
                if arrays is None:
                    continue
                val_hours, test_hours = val_test_hours(arrays, cfg)
                for seed in args.seeds:
                    for method_dir, method_name in [
                        ("M2", "M2_target_only"),
                        ("M3", "M3_pretrain_ft"),
                    ]:
                        ckpt = phase2 / target_id / model_type / f"k{k}" / f"seed{seed}" / method_dir / "model.pt"
                        if not ckpt.exists():
                            continue
                        model = make_model(model_type, len(feature_cols), cfg.data.window_size, cfg).to(device)
                        state = torch.load(ckpt, map_location=device)
                        model.load_state_dict(state)
                        val_ldr = make_dataloader(arrays["X_val"], arrays["y_val_s"],
                                                  cfg.train.batch_size, shuffle=False)
                        test_ldr = make_dataloader(arrays["X_test"], arrays["y_test_s"],
                                                   cfg.train.batch_size, shuffle=False)
                        yv, pv = predict(model, val_ldr, device, y_scaler=arrays["y_scaler"])
                        yt, pt = predict(model, test_ldr, device, y_scaler=arrays["y_scaler"])
                        del model

                        val_err = np.abs(yv - pv)
                        test_err = np.abs(yt - pt)
                        val_parts, test_parts = build_scores(
                            val_err, test_err, arrays["y_val"], yt,
                            val_hours, test_hours, cfg
                        )
                        for detector in DETECTORS:
                            score, pred, calibration = detector_output(detector, val_parts, test_parts)
                            row = {
                                "target": target_id,
                                "model": model_type,
                                "k": k,
                                "seed": seed,
                                "method": method_name,
                                "detector": detector,
                                "calibration": calibration,
                            }
                            row.update(metrics_from_prediction(arrays["labels"], score, pred))
                            add_type_recalls(row, arrays["labels"], pred, arrays["types"])
                            rows.append(row)
                        print(f"[event-detector] {target_id} {model_type} k{k} seed{seed} {method_name}")

    df = pd.DataFrame(rows)
    detail_path = output / "table_event_detector.csv"
    df.to_csv(detail_path, index=False)

    if not df.empty:
        summary_cols = [
            "Precision", "Recall", "F1", "AUROC", "AUPRC", "FAR",
            "EventPrecision", "EventRecall", "EventF1", "EventDelay",
        ]
        summary = df.groupby(["model", "k", "method", "detector"])[summary_cols].mean().reset_index()
        summary.to_csv(output / "table_event_detector_summary.csv", index=False)

        pairs = []
        for key, g in df.groupby(["target", "model", "k", "seed", "detector"]):
            m2 = g[g["method"] == "M2_target_only"]
            m3 = g[g["method"] == "M3_pretrain_ft"]
            if len(m2) and len(m3):
                m2 = m2.iloc[0]
                m3 = m3.iloc[0]
                pairs.append({
                    "target": key[0], "model": key[1], "k": key[2],
                    "seed": key[3], "detector": key[4],
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
        pair_df.to_csv(output / "table_event_detector_m2_m3_pairs.csv", index=False)
        if not pair_df.empty:
            claim = pair_df.groupby(["model", "k", "detector"]).agg(
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
            claim.to_csv(output / "table_event_detector_claim_summary.csv", index=False)
            print(claim.sort_values(
                ["model", "k", "mean_delta_EventF1", "mean_delta_AUPRC"],
                ascending=[True, True, False, False],
            ).to_string(index=False))

    manifest = {
        "detectors": DETECTORS,
        "calibration": "clean validation residuals only; injected test labels are never used for threshold selection",
        "outputs": [
            str(detail_path),
            str(output / "table_event_detector_summary.csv"),
            str(output / "table_event_detector_claim_summary.csv"),
        ],
    }
    (output / "event_detector_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[event-detector] saved {detail_path} rows={len(df)}")


if __name__ == "__main__":
    main()
