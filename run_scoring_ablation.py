#!/usr/bin/env python
"""Run anomaly-scoring ablation on completed Phase 2 checkpoints.

This script does not retrain models. It loads saved M2/M3 checkpoints,
reconstructs validation and injected-test windows, calibrates thresholds on
clean validation data, and evaluates detection on controlled injected labels.
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

from src.anomaly import rolling_mad_scores
from src.anomaly_injection import inject_anomalies
from src.config import Config
from src.phase2 import FEATURE_COLS, make_model
from src.train import (
    apply_scaler,
    apply_scaler_1d,
    build_windows,
    fit_scaler,
    fit_scaler_1d,
    make_dataloader,
    predict,
)


SCORERS = [
    "residual_q95",
    "residual_q975",
    "residual_q99",
    "zscore_q975",
    "rolling_mad_q975",
    "ewma_q975",
    "cusum_q975",
    "flatline_q975",
    "hybrid_q975",
]


def quantile_threshold(scores: np.ndarray, q: float) -> float:
    return float(np.quantile(scores[np.isfinite(scores)], q))


def zscore_scores(errors: np.ndarray, ref_errors: np.ndarray) -> np.ndarray:
    mu = float(np.mean(ref_errors))
    sigma = float(np.std(ref_errors))
    if sigma < 1e-8:
        sigma = 1.0
    return (errors - mu) / sigma


def ewma_scores(errors: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    baseline = float(np.median(errors))
    out = np.zeros_like(errors, dtype=float)
    state = 0.0
    for i, e in enumerate(errors):
        state = alpha * max(0.0, float(e) - baseline) + (1 - alpha) * state
        out[i] = state
    return out


def cusum_scores(errors: np.ndarray, ref_errors: np.ndarray, k: float = 0.5) -> np.ndarray:
    mu = float(np.mean(ref_errors))
    sigma = float(np.std(ref_errors))
    drift = mu + k * sigma
    out = np.zeros_like(errors, dtype=float)
    state = 0.0
    for i, e in enumerate(errors):
        state = max(0.0, state + float(e) - drift)
        out[i] = state
    return out


def flatline_scores(y: np.ndarray, window: int = 6) -> np.ndarray:
    """Score low-variation runs in the observed series.

    Higher means more likely stuck/flatline. Uses a rolling inverse std of
    first differences, so constant stretches become high-scoring.
    """
    y = np.asarray(y, dtype=float)
    diffs = np.abs(np.diff(y, prepend=y[0]))
    out = np.zeros_like(y, dtype=float)
    global_scale = max(float(np.std(diffs)), 1e-6)
    for i in range(len(y)):
        start = max(0, i - window + 1)
        local = diffs[start:i + 1]
        local_mean = float(np.mean(local))
        out[i] = global_scale / (local_mean + 1e-6)
    return np.clip(out, 0, 1e6)


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores > threshold).astype(int)
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
    return out


def contiguous_events(labels: np.ndarray) -> list[tuple[int, int]]:
    events = []
    start = None
    for i, v in enumerate(labels.astype(int)):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(labels) - 1))
    return events


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def event_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores > threshold).astype(int)
    true_events = contiguous_events(labels)
    pred_events = contiguous_events(pred)

    detected = 0
    delays = []
    for te in true_events:
        hits = [pe for pe in pred_events if overlaps(te, pe)]
        if hits:
            detected += 1
            first_hit = min(max(pe[0], te[0]) for pe in hits)
            delays.append(max(0, first_hit - te[0]))

    matched_pred = 0
    for pe in pred_events:
        if any(overlaps(pe, te) for te in true_events):
            matched_pred += 1

    event_recall = detected / max(len(true_events), 1)
    event_precision = matched_pred / max(len(pred_events), 1)
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if (event_precision + event_recall) > 0 else 0.0
    )
    return {
        "EventPrecision": float(event_precision),
        "EventRecall": float(event_recall),
        "EventF1": float(event_f1),
        "EventDelay": float(np.mean(delays)) if delays else float("nan"),
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
    }


def add_type_recalls(row: dict, labels: np.ndarray, scores: np.ndarray,
                     threshold: float, types: np.ndarray):
    pred = (scores > threshold).astype(int)
    for anomaly_type in sorted(set(types) - {"normal"}):
        mask = types == anomaly_type
        if mask.sum() == 0:
            continue
        row[f"Recall_{anomaly_type}"] = float(
            recall_score(labels[mask], pred[mask], zero_division=0)
        )


def score_bundle(name: str, val_err: np.ndarray, test_err: np.ndarray,
                 val_y: np.ndarray, test_y: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, float]:
    if name.startswith("residual_q"):
        q = {"residual_q95": 0.95, "residual_q975": 0.975, "residual_q99": 0.99}[name]
        val_score = val_err
        test_score = test_err
        threshold = quantile_threshold(val_score, q)
    elif name == "zscore_q975":
        val_score = zscore_scores(val_err, val_err)
        test_score = zscore_scores(test_err, val_err)
        threshold = quantile_threshold(val_score, 0.975)
    elif name == "rolling_mad_q975":
        val_score = rolling_mad_scores(val_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
        test_score = rolling_mad_scores(test_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
        threshold = quantile_threshold(val_score, 0.975)
    elif name == "ewma_q975":
        val_score = ewma_scores(val_err)
        test_score = ewma_scores(test_err)
        threshold = quantile_threshold(val_score, 0.975)
    elif name == "cusum_q975":
        val_score = cusum_scores(val_err, val_err)
        test_score = cusum_scores(test_err, val_err)
        threshold = quantile_threshold(val_score, 0.975)
    elif name == "flatline_q975":
        val_score = flatline_scores(val_y)
        test_score = flatline_scores(test_y)
        threshold = quantile_threshold(val_score, 0.975)
    elif name == "hybrid_q975":
        parts_val = [
            zscore_scores(val_err, val_err),
            rolling_mad_scores(val_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon),
            ewma_scores(val_err),
            cusum_scores(val_err, val_err),
            flatline_scores(val_y),
        ]
        parts_test = [
            zscore_scores(test_err, val_err),
            rolling_mad_scores(test_err, cfg.anomaly.mad_window, cfg.anomaly.epsilon),
            ewma_scores(test_err),
            cusum_scores(test_err, val_err),
            flatline_scores(test_y),
        ]
        norm_val, norm_test = [], []
        for v, t in zip(parts_val, parts_test):
            scale = quantile_threshold(v, 0.975)
            if abs(scale) < 1e-8:
                scale = 1.0
            norm_val.append(v / scale)
            norm_test.append(t / scale)
        val_score = np.max(np.vstack(norm_val), axis=0)
        test_score = np.max(np.vstack(norm_test), axis=0)
        threshold = 1.0
    else:
        raise ValueError(f"Unknown scorer: {name}")
    return val_score, test_score, threshold


def prepare_arrays(data: pd.DataFrame, target_id: str, k: int,
                   feature_cols: list[str], cfg: Config):
    tgt = data[data["building_id"] == target_id].sort_values("timestamp")
    n = len(tgt)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_full = tgt.iloc[:train_end]
    val = tgt.iloc[train_end:val_end]
    test = tgt.iloc[val_end:].copy()

    if k > 0:
        train_few = train_full.iloc[:k * 24]
        X_fit, y_fit = build_windows(train_few, feature_cols, "energy",
                                     cfg.data.window_size, cfg.data.horizon)
        if len(X_fit) == 0:
            return None
    else:
        X_fit, y_fit = build_windows(train_full, feature_cols, "energy",
                                     cfg.data.window_size, cfg.data.horizon)
        if len(X_fit) == 0:
            return None

    x_mean, x_std = fit_scaler(X_fit)
    y_mean, y_std = fit_scaler_1d(y_fit)

    X_val, y_val = build_windows(val, feature_cols, "energy",
                                 cfg.data.window_size, cfg.data.horizon)
    injected_energy, point_labels, point_types = inject_anomalies(
        test["energy"].values, injection_rate=0.05, seed=cfg.seed, return_types=True
    )
    test_inj = test.copy()
    test_inj["energy"] = injected_energy
    X_test, y_test = build_windows(test_inj, feature_cols, "energy",
                                   cfg.data.window_size, cfg.data.horizon)
    offsets = np.arange(len(y_test)) + cfg.data.window_size + cfg.data.horizon - 1
    labels = point_labels[offsets]
    types = point_types[offsets]

    return {
        "X_val": apply_scaler(X_val, x_mean, x_std),
        "y_val_s": apply_scaler_1d(y_val, y_mean, y_std),
        "y_val": y_val,
        "X_test": apply_scaler(X_test, x_mean, x_std),
        "y_test_s": apply_scaler_1d(y_test, y_mean, y_std),
        "y_test": y_test,
        "labels": labels,
        "types": types,
        "y_scaler": (y_mean, y_std),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    ap.add_argument("--phase2-dir", default="results/phase2_paper")
    ap.add_argument("--output-dir", default="results/phase2_paper/aggregate")
    ap.add_argument("--models", nargs="+", default=["lstm", "dlinear"])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    phase2 = Path(args.phase2_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for target_id in sorted(data["building_id"].unique()):
        for model_type in args.models:
            for k in args.k:
                arrays = prepare_arrays(data, target_id, k, feature_cols, cfg)
                if arrays is None:
                    continue
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
                        for scorer in SCORERS:
                            _, test_score, threshold = score_bundle(
                                scorer, val_err, test_err, arrays["y_val"], yt, cfg
                            )
                            row = {
                                "target": target_id,
                                "model": model_type,
                                "k": k,
                                "seed": seed,
                                "method": method_name,
                                "scorer": scorer,
                                "threshold": threshold,
                            }
                            row.update(detection_metrics(arrays["labels"], test_score, threshold))
                            row.update(event_metrics(arrays["labels"], test_score, threshold))
                            add_type_recalls(row, arrays["labels"], test_score,
                                             threshold, arrays["types"])
                            rows.append(row)
                        print(f"[scoring] {target_id} {model_type} k{k} seed{seed} {method_name}")

    df = pd.DataFrame(rows)
    detail_path = output / "table_scoring_ablation.csv"
    df.to_csv(detail_path, index=False)

    if not df.empty:
        summary = df.groupby(["model", "k", "method", "scorer"])[
            ["Precision", "Recall", "F1", "AUROC", "AUPRC", "FAR",
             "EventPrecision", "EventRecall", "EventF1", "EventDelay"]
        ].mean().reset_index()
        summary.to_csv(output / "table_scoring_ablation_summary.csv", index=False)

        pairs = []
        for key, g in df.groupby(["target", "model", "k", "seed", "scorer"]):
            m2 = g[g["method"] == "M2_target_only"]
            m3 = g[g["method"] == "M3_pretrain_ft"]
            if len(m2) and len(m3):
                m2 = m2.iloc[0]
                m3 = m3.iloc[0]
                pairs.append({
                    "target": key[0], "model": key[1], "k": key[2],
                    "seed": key[3], "scorer": key[4],
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
        pair_df.to_csv(output / "table_scoring_ablation_m2_m3_pairs.csv", index=False)
        if not pair_df.empty:
            claim = pair_df.groupby(["model", "k", "scorer"]).agg(
                n=("target", "count"),
                F1_win_rate=("M3_F1_win", "mean"),
                AUPRC_win_rate=("M3_AUPRC_win", "mean"),
                FAR_win_rate=("M3_FAR_win", "mean"),
                mean_delta_F1=("delta_F1", "mean"),
                mean_delta_AUPRC=("delta_AUPRC", "mean"),
                mean_delta_FAR=("delta_FAR", "mean"),
                EventF1_win_rate=("M3_EventF1_win", "mean"),
                mean_delta_EventF1=("delta_EventF1", "mean"),
            ).reset_index()
            claim.to_csv(output / "table_scoring_ablation_claim_summary.csv", index=False)
            print(claim.sort_values(["model", "k", "mean_delta_AUPRC"],
                                    ascending=[True, True, False]).to_string(index=False))

    print(f"[scoring] saved {detail_path} rows={len(df)}")


if __name__ == "__main__":
    main()
