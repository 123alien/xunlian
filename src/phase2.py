"""Phase 2: Full experiment matrix for paper.

Covers:
- 12 buildings, leave-one-out cross-validation
- Models: LSTM, DLinear, PatchTST
- k ∈ {0, 1, 3, 7, 14} days few-shot
- Methods: M1 (source-only), M2 (target-only few-shot), M3 (pretrain+FT)
- Seeds: 42, 43, 44, 45, 46
- Anomaly injection at 5%
- Unsupervised AD baselines: Isolation Forest, LOF, LSTM-AE

Output structure:
  results/phase2/
  ├── <target_building>/<model>/k<k>/seed<s>/<method>/
  ├── <target_building>/baselines/
  ├── <target_building>/anomaly_injection/
  └── aggregate/
"""
import sys
import json
import yaml
from pathlib import Path
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.models import LSTMForecaster, DLinear, PatchTSTForecaster
from src.train import (
    build_windows, fit_scaler, apply_scaler, make_dataloader,
    train_model, pretrain, fine_tune, predict, save_run,
    fit_scaler_1d, apply_scaler_1d,
)
from src.anomaly import rolling_mad_scores
from src.metrics import compute_all
from src.anomaly_injection import inject_anomalies
from src.baselines import (
    run_isolation_forest, run_lof,
    LSTMAutoencoder, train_lstm_ae, lstm_ae_anomaly_scores,
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["energy", "hour", "day_of_week", "is_weekend", "month"]
K_VALUES = [0, 1, 3, 7, 14]
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["lstm", "dlinear", "patchtst"]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(model_type: str, input_dim: int, window_size: int,
               config: Config) -> nn.Module:
    if model_type == "lstm":
        return LSTMForecaster(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        )
    elif model_type == "dlinear":
        return DLinear(
            window_size=window_size,
            input_dim=input_dim,
            horizon=1,
        )
    elif model_type == "patchtst":
        return PatchTSTForecaster(
            window_size=window_size,
            input_dim=input_dim,
            patch_len=6,
            stride=3,
            d_model=64,
            n_heads=4,
            num_layers=2,
            dropout=config.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")


def run_phase2(config_path: str = "configs/default.yaml",
               data_path: str = "data/processed/bdg2_electricity_hourly.csv",
               output_dir: str = "results/phase2",
               n_buildings: int = 12,
               building_manifest: str | None = None,
               k_values: list[int] | None = None,
               seeds: list[int] | None = None,
               models: list[str] | None = None,
               skip_baselines: bool = False,
               skip_injection: bool = False):
    """Run full Phase 2 experiment matrix."""

    cfg = Config.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase2] Device: {device}")

    ks = k_values if k_values is not None else K_VALUES
    sds = seeds if seeds is not None else SEEDS
    mdls = models if models is not None else MODELS

    # Load data
    data = pd.read_csv(data_path, parse_dates=["timestamp"])
    if building_manifest:
        manifest = pd.read_csv(building_manifest)
        building_ids = manifest["building_id"].dropna().astype(str).tolist()[:n_buildings]
    else:
        building_ids = sorted(data["building_id"].unique())[:n_buildings]
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    n_features = len(feature_cols)
    print(f"[Phase2] Buildings: {building_ids}")
    print(f"[Phase2] Features: {feature_cols}")
    print(f"[Phase2] k values: {ks}, seeds: {sds}, models: {mdls}")

    all_few_shot_rows = []
    all_injection_rows = []
    all_baseline_rows = []
    out_root = Path(output_dir)

    # ================================================================
    # Main loop: for each target building
    # ================================================================
    for tgt_idx, target_id in enumerate(building_ids):
        print(f"\n{'='*60}")
        print(f"[Phase2] Target {tgt_idx+1}/{len(building_ids)}: {target_id}")
        print(f"{'='*60}")

        source_ids = [b for b in building_ids if b != target_id]
        if len(source_ids) < 2:
            print("  SKIP: need at least 2 source buildings")
            continue

        # ---- Prepare target data (shared across models/methods) ----
        tgt_data = data[data["building_id"] == target_id].sort_values("timestamp")
        n_tgt = len(tgt_data)
        train_end = int(n_tgt * 0.6)
        val_end = int(n_tgt * (0.6 + 0.2))
        tgt_train_full = tgt_data.iloc[:train_end]
        tgt_val = tgt_data.iloc[train_end:val_end]
        tgt_test = tgt_data.iloc[val_end:]

        # Test windows (shared)
        X_test, y_test = build_windows(tgt_test, feature_cols, "energy",
                                       cfg.data.window_size, cfg.data.horizon)
        if len(X_test) < 50:
            print(f"  SKIP: insufficient test data ({len(X_test)} windows)")
            continue

        print(f"  Target: {n_tgt} rows, {len(X_test)} test windows")

        # ---- Source loaders (shared across k, but re-made per model) ----
        # We'll rebuild source loaders inside the model loop for cleanliness.

        # ---- Each model ----
        for model_type in mdls:
            print(f"\n  --- Model: {model_type} ---")

            # Prepare source loaders (with y-scaling)
            source_loaders, source_y_scalers = _prepare_source_loaders(
                data, source_ids, feature_cols, cfg, device
            )
            if not source_loaders:
                print("    SKIP: no valid source loaders")
                continue

            for k in ks:
                k_hours = k * 24
                print(f"    k={k} days ({k_hours}h)")

                # Prepare few-shot data (shared across seeds)
                if k > 0:
                    tgt_train_few = tgt_train_full.iloc[:k_hours]
                    X_few, y_few = build_windows(tgt_train_few, feature_cols, "energy",
                                                 cfg.data.window_size, cfg.data.horizon)
                    if len(X_few) < 10:
                        print(f"      SKIP: insufficient few-shot windows ({len(X_few)})")
                        continue
                else:
                    X_few, y_few = None, None

                X_val, y_val = build_windows(tgt_val, feature_cols, "energy",
                                             cfg.data.window_size, cfg.data.horizon)

                # Scale X. For k=0 there is no target few-shot window; use the
                # allowed target training period rather than the test set to
                # avoid leaking target test distribution into zero-shot runs.
                if X_few is not None and len(X_few) > 0:
                    few_mean, few_std = fit_scaler(X_few)
                else:
                    X_train_full, _ = build_windows(tgt_train_full, feature_cols, "energy",
                                                    cfg.data.window_size, cfg.data.horizon)
                    if len(X_train_full) == 0:
                        print("      SKIP: insufficient target train windows for k=0 scaling")
                        continue
                    few_mean, few_std = fit_scaler(X_train_full)

                X_val_s = apply_scaler(X_val, few_mean, few_std)
                X_test_s = apply_scaler(X_test, few_mean, few_std)

                if X_few is not None:
                    X_few_s = apply_scaler(X_few, few_mean, few_std)

                # Scale Y on target data
                if X_few is not None and len(X_few) > 0:
                    tgt_y_mean, tgt_y_std = fit_scaler_1d(y_few)
                else:
                    tgt_y_full = tgt_train_full["energy"].values
                    tgt_y_mean, tgt_y_std = fit_scaler_1d(tgt_y_full)

                if X_few is not None:
                    y_few_s = apply_scaler_1d(y_few, tgt_y_mean, tgt_y_std)
                else:
                    y_few_s = None
                y_val_s = apply_scaler_1d(y_val, tgt_y_mean, tgt_y_std)
                y_test_s = apply_scaler_1d(y_test, tgt_y_mean, tgt_y_std)
                tgt_y_scaler = (tgt_y_mean, tgt_y_std)

                for seed in sds:
                    print(f"      seed={seed}")

                    # -- M2: Target-only few-shot (only if k > 0) --
                    if k > 0:
                        m2_dir = out_root / target_id / model_type / f"k{k}" / f"seed{seed}" / "M2"
                        if not (m2_dir / "metrics.json").exists():
                            set_seed(seed)
                            model_m2 = make_model(model_type, n_features,
                                                  cfg.data.window_size, cfg).to(device)
                            few_bs = max(4, min(cfg.train.batch_size, len(X_few_s) // 4))
                            few_train_ldr = make_dataloader(X_few_s, y_few_s, few_bs, shuffle=True)
                            few_val_ldr = make_dataloader(X_val_s, y_val_s, few_bs, shuffle=False)
                            test_ldr = make_dataloader(X_test_s, y_test_s, cfg.train.batch_size, shuffle=False)

                            history_m2 = train_model(model_m2, few_train_ldr, few_val_ldr, cfg, device, verbose=False)
                            yt_m2, yp_m2 = predict(model_m2, test_ldr, device, y_scaler=tgt_y_scaler)
                            err_m2 = np.abs(yt_m2 - yp_m2)
                            sc_m2 = rolling_mad_scores(err_m2, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                            met_m2 = compute_all(yt_m2, yp_m2, sc_m2)
                            met_m2.update({"method": "M2_target_only", "model": model_type,
                                           "target": target_id, "k": k, "seed": seed})
                            save_run(str(m2_dir), cfg, met_m2, predictions=(yt_m2, yp_m2),
                                     anomaly_scores=sc_m2, history=history_m2,
                                     model_state=model_m2.state_dict(),
                                     manifest={
                                         "method": "M2_target_only",
                                         "model": model_type,
                                         "target": target_id,
                                         "k": k,
                                         "seed": seed,
                                         "x_scaler_policy": "target_few_shot",
                                         "y_scaler_policy": "target_few_shot",
                                     })
                            del model_m2

                    # -- M3: Pretrain + Fine-tune (or just pretrain if k=0) --
                    m3_dir = out_root / target_id / model_type / f"k{k}" / f"seed{seed}" / "M3"
                    if not (m3_dir / "metrics.json").exists():
                        set_seed(seed)
                        model_m3 = make_model(model_type, n_features,
                                              cfg.data.window_size, cfg).to(device)

                        # Pretrain
                        hist_pt = pretrain(model_m3, source_loaders, cfg, device, verbose=False)

                        # Fine-tune (if k>0)
                        if k > 0:
                            few_bs = max(4, min(cfg.train.batch_size, len(X_few_s) // 4))
                            few_train_ldr = make_dataloader(X_few_s, y_few_s, few_bs, shuffle=True)
                            few_val_ldr = make_dataloader(X_val_s, y_val_s, few_bs, shuffle=False)
                            hist_ft = fine_tune(model_m3, few_train_ldr, few_val_ldr, cfg, device, verbose=False)
                        else:
                            hist_ft = {}

                        test_ldr = make_dataloader(X_test_s, y_test_s, cfg.train.batch_size, shuffle=False)
                        yt_m3, yp_m3 = predict(model_m3, test_ldr, device, y_scaler=tgt_y_scaler)
                        err_m3 = np.abs(yt_m3 - yp_m3)
                        sc_m3 = rolling_mad_scores(err_m3, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                        met_m3 = compute_all(yt_m3, yp_m3, sc_m3)
                        met_m3.update({"method": "M3_pretrain_ft", "model": model_type,
                                       "target": target_id, "k": k, "seed": seed})
                        save_run(str(m3_dir), cfg, met_m3, predictions=(yt_m3, yp_m3),
                                 anomaly_scores=sc_m3,
                                 history={"pretrain": hist_pt, "finetune": hist_ft if k > 0 else {}},
                                 model_state=model_m3.state_dict(),
                                 manifest={
                                     "method": "M3_pretrain_ft",
                                     "model": model_type,
                                     "target": target_id,
                                     "k": k,
                                     "seed": seed,
                                     "x_scaler_policy": "target_few_shot" if k > 0 else "target_train_full",
                                     "y_scaler_policy": "target_few_shot" if k > 0 else "target_train_full",
                                 })
                        del model_m3

                    # -- M1: Source-only (k=0 only, equivalent to M3 with k=0) --
                    if k == 0:
                        m1_dir = out_root / target_id / model_type / f"k0" / f"seed{seed}" / "M1"
                        if not (m1_dir / "metrics.json").exists():
                            set_seed(seed)
                            model_m1 = make_model(model_type, n_features,
                                                  cfg.data.window_size, cfg).to(device)
                            hist_m1 = pretrain(model_m1, source_loaders, cfg, device, verbose=False)
                            test_ldr = make_dataloader(X_test_s, y_test_s, cfg.train.batch_size, shuffle=False)
                            yt_m1, yp_m1 = predict(model_m1, test_ldr, device, y_scaler=tgt_y_scaler)
                            err_m1 = np.abs(yt_m1 - yp_m1)
                            sc_m1 = rolling_mad_scores(err_m1, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                            met_m1 = compute_all(yt_m1, yp_m1, sc_m1)
                            met_m1.update({"method": "M1_source_only", "model": model_type,
                                           "target": target_id, "k": 0, "seed": seed})
                            save_run(str(m1_dir), cfg, met_m1, predictions=(yt_m1, yp_m1),
                                     anomaly_scores=sc_m1, history={"pretrain": hist_m1},
                                     model_state=model_m1.state_dict(),
                                     manifest={
                                         "method": "M1_source_only",
                                         "model": model_type,
                                         "target": target_id,
                                         "k": 0,
                                         "seed": seed,
                                         "x_scaler_policy": "target_train_full",
                                         "y_scaler_policy": "target_train_full",
                                     })
                            del model_m1

                # Free CUDA memory between k values
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # ---- Unsupervised AD Baselines ----
        if not skip_baselines:
            print(f"\n  --- AD Baselines ---")
            _run_ad_baselines(data, building_ids, target_id, feature_cols, cfg, device,
                              out_root, all_baseline_rows)

        # ---- Anomaly Injection ----
        if not skip_injection:
            print(f"\n  --- Anomaly Injection (5%) ---")
            _run_anomaly_injection(data, building_ids, target_id, feature_cols, cfg, device,
                                   out_root, mdls, ks, sds, X_test, y_test, all_injection_rows)

    # ================================================================
    # Aggregate
    # ================================================================
    _aggregate_results(out_root, building_ids, mdls, ks, sds)
    print(f"\n[Phase2] Done. Results in {out_root}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_source_loaders(data, source_ids, feature_cols, cfg, device):
    """Build (train_loader, val_loader) for each source building.

    Returns (loaders, y_scalers) where:
      loaders: list of (train_ldr, val_ldr) with scaled X and scaled y
      y_scalers: list of (y_mean, y_std) per source building
    """
    loaders = []
    y_scalers = []
    for src_id in source_ids:
        src_data = data[data["building_id"] == src_id].sort_values("timestamp")
        n_src = len(src_data)
        if n_src < cfg.data.window_size * 3:
            continue
        src_train = src_data.iloc[:int(n_src * 0.6)]
        src_val = src_data.iloc[int(n_src * 0.6):int(n_src * 0.8)]

        X_tr, y_tr = build_windows(src_train, feature_cols, "energy",
                                   cfg.data.window_size, cfg.data.horizon)
        X_v, y_v = build_windows(src_val, feature_cols, "energy",
                                 cfg.data.window_size, cfg.data.horizon)
        if len(X_tr) < 10:
            continue

        mean_s, std_s = fit_scaler(X_tr)
        y_mean, y_std = fit_scaler_1d(y_tr)
        y_tr_s = apply_scaler_1d(y_tr, y_mean, y_std)
        y_v_s = apply_scaler_1d(y_v, y_mean, y_std)
        train_ldr = make_dataloader(apply_scaler(X_tr, mean_s, std_s), y_tr_s,
                                    cfg.train.batch_size, shuffle=True)
        val_ldr = make_dataloader(apply_scaler(X_v, mean_s, std_s), y_v_s,
                                  cfg.train.batch_size, shuffle=False)
        loaders.append((train_ldr, val_ldr))
        y_scalers.append((y_mean, y_std))
    return loaders, y_scalers


def _run_ad_baselines(data, building_ids, target_id, feature_cols, cfg, device,
                      out_root, all_baseline_rows):
    """Run Isolation Forest, LOF, LSTM-AE for one target building."""
    source_ids = [b for b in building_ids if b != target_id]

    # Prepare source feature matrix (all windows from all sources)
    X_src_list, y_src_list = [], []
    for src_id in source_ids:
        src_data = data[data["building_id"] == src_id].sort_values("timestamp")
        n_src = len(src_data)
        src_train = src_data.iloc[:int(n_src * 0.6)]
        X_s, y_s = build_windows(src_train, feature_cols, "energy",
                                 cfg.data.window_size, cfg.data.horizon)
        if len(X_s) > 0:
            X_src_list.append(X_s)
            y_src_list.append(y_s)
    if not X_src_list:
        return
    X_src_all = np.concatenate(X_src_list)
    y_src_all = np.concatenate(y_src_list)

    # Target test data with controlled injected anomalies.
    tgt_data = data[data["building_id"] == target_id].sort_values("timestamp")
    n_tgt = len(tgt_data)
    tgt_test = tgt_data.iloc[int(n_tgt * 0.8):].copy()
    injected_energy, point_labels, point_types = inject_anomalies(
        tgt_test["energy"].values, injection_rate=0.05, seed=cfg.seed, return_types=True
    )
    tgt_test_inj = tgt_test.copy()
    tgt_test_inj["energy"] = injected_energy
    X_tgt, y_tgt = build_windows(
        tgt_test_inj, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon
    )
    target_offsets = np.arange(len(y_tgt)) + cfg.data.window_size + cfg.data.horizon - 1
    labels_tgt = point_labels[target_offsets]
    types_tgt = point_types[target_offsets]

    if len(X_tgt) < 50:
        return

    base_dir = out_root / target_id / "baselines"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Isolation Forest
    if_scores = run_isolation_forest(X_src_all, X_tgt, seed=cfg.seed)
    if_metrics = _ad_detection_metrics(labels_tgt, if_scores, cfg.anomaly.threshold)
    if_metrics.update({"method": "IsolationForest", "target": target_id,
                       "injection_rate": 0.05})
    _add_type_recalls(if_metrics, labels_tgt, if_scores, types_tgt, cfg.anomaly.threshold)
    save_run(str(base_dir / "isolation_forest"), cfg, if_metrics, anomaly_scores=if_scores)
    all_baseline_rows.append(if_metrics)

    # LOF
    lof_scores = run_lof(X_src_all[:len(X_src_all)//3], X_tgt)  # subsample for speed
    lof_metrics = _ad_detection_metrics(labels_tgt, lof_scores, cfg.anomaly.threshold)
    lof_metrics.update({"method": "LOF", "target": target_id,
                        "injection_rate": 0.05})
    _add_type_recalls(lof_metrics, labels_tgt, lof_scores, types_tgt, cfg.anomaly.threshold)
    save_run(str(base_dir / "lof"), cfg, lof_metrics, anomaly_scores=lof_scores)
    all_baseline_rows.append(lof_metrics)

    # LSTM-AE
    try:
        mean_s, std_s = fit_scaler(X_src_all)
        X_src_s = apply_scaler(X_src_all, mean_s, std_s)
        X_tgt_s = apply_scaler(X_tgt, mean_s, std_s)

        src_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_src_s).float(),
            torch.from_numpy(X_src_s).float()  # autoencoder: input = target
        )
        src_ldr = DataLoader(src_ds, batch_size=cfg.train.batch_size, shuffle=True)
        val_size = min(len(src_ds) // 5, len(X_tgt_s))
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_src_s[-val_size:]).float(),
            torch.from_numpy(X_src_s[-val_size:]).float(),
        )
        val_ldr = DataLoader(val_ds, batch_size=cfg.train.batch_size)
        tgt_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tgt_s).float(),
            torch.from_numpy(X_tgt_s).float(),
        )
        tgt_ldr = DataLoader(tgt_ds, batch_size=cfg.train.batch_size)

        ae = LSTMAutoencoder(input_dim=len(feature_cols)).to(device)
        train_lstm_ae(ae, src_ldr, val_ldr, epochs=30, device=device, verbose=False)
        ae_scores = lstm_ae_anomaly_scores(ae, tgt_ldr, device)
        ae_metrics = _ad_detection_metrics(labels_tgt, ae_scores, cfg.anomaly.threshold)
        ae_metrics.update({"method": "LSTM_AE", "target": target_id,
                           "injection_rate": 0.05})
        _add_type_recalls(ae_metrics, labels_tgt, ae_scores, types_tgt, cfg.anomaly.threshold)
        save_run(str(base_dir / "lstm_ae"), cfg, ae_metrics, anomaly_scores=ae_scores)
        all_baseline_rows.append(ae_metrics)
        del ae
    except Exception as e:
        print(f"    LSTM-AE failed: {e}")


def _ad_detection_metrics(labels: np.ndarray, scores: np.ndarray,
                          threshold: float) -> dict:
    pred = (scores > threshold).astype(int)
    metrics = {
        "Precision": float(precision_score(labels, pred, zero_division=0)),
        "Recall": float(recall_score(labels, pred, zero_division=0)),
        "F1": float(f1_score(labels, pred, zero_division=0)),
        "FAR": float(pred[labels == 0].sum() / max((labels == 0).sum(), 1)),
    }
    try:
        metrics["AUROC"] = float(roc_auc_score(labels, scores))
        metrics["AUPRC"] = float(average_precision_score(labels, scores))
    except ValueError:
        metrics["AUROC"] = float("nan")
        metrics["AUPRC"] = float("nan")
    return metrics


def _add_type_recalls(metrics: dict, labels: np.ndarray, scores: np.ndarray,
                      types: np.ndarray, threshold: float):
    pred = (scores > threshold).astype(int)
    for anomaly_type in sorted(set(types) - {"normal"}):
        mask = types == anomaly_type
        if mask.sum() == 0:
            continue
        metrics[f"Recall_{anomaly_type}"] = float(
            recall_score(labels[mask], pred[mask], zero_division=0)
        )


def _run_anomaly_injection(data, building_ids, target_id, feature_cols, cfg, device,
                           out_root, mdls, ks, sds, X_test, y_test, all_injection_rows):
    """Run anomaly injection evaluation for one target building."""
    inj_dir = out_root / target_id / "anomaly_injection"
    inj_dir.mkdir(parents=True, exist_ok=True)

    tgt_data = data[data["building_id"] == target_id].sort_values("timestamp")
    n_tgt = len(tgt_data)
    train_end = int(n_tgt * 0.6)
    val_end = int(n_tgt * 0.8)
    tgt_train_full = tgt_data.iloc[:train_end]
    tgt_test = tgt_data.iloc[val_end:].copy()

    # Inject anomalies into the observed test series, then rebuild windows so
    # both model inputs and target values reflect the injected test sequence.
    injected_energy, point_labels, point_types = inject_anomalies(
        tgt_test["energy"].values, injection_rate=0.05, seed=cfg.seed, return_types=True
    )
    tgt_test_inj = tgt_test.copy()
    tgt_test_inj["energy"] = injected_energy
    X_test_inj, y_test_inj = build_windows(
        tgt_test_inj, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon
    )
    target_offsets = np.arange(len(y_test_inj)) + cfg.data.window_size + cfg.data.horizon - 1
    inj_labels = point_labels[target_offsets]
    inj_types = point_types[target_offsets]

    # Keep injection evaluation bounded by using one representative seed.
    # Main forecasting/stability tables still use all configured seeds.
    rep_seed = 42

    for model_type in mdls:
        for k in ks:
            if k == 0:
                continue  # k=0 has no fine-tuning, skip injection eval for it
            m3_dir = out_root / target_id / model_type / f"k{k}" / f"seed{rep_seed}" / "M3"
            m2_dir = out_root / target_id / model_type / f"k{k}" / f"seed{rep_seed}" / "M2"

            for method_dir, method_name in [(m3_dir, "M3_pretrain_ft"), (m2_dir, "M2_target_only")]:
                model_file = method_dir / "model.pt"
                if not model_file.exists():
                    print(f"    SKIP injection: missing checkpoint {model_file}")
                    continue

                if k > 0:
                    tgt_train_few = tgt_train_full.iloc[:k * 24]
                    X_few, y_few = build_windows(
                        tgt_train_few, feature_cols, "energy",
                        cfg.data.window_size, cfg.data.horizon
                    )
                    if len(X_few) == 0:
                        continue
                    x_mean, x_std = fit_scaler(X_few)
                    y_mean, y_std = fit_scaler_1d(y_few)
                else:
                    X_train_full, y_train_full = build_windows(
                        tgt_train_full, feature_cols, "energy",
                        cfg.data.window_size, cfg.data.horizon
                    )
                    if len(X_train_full) == 0:
                        continue
                    x_mean, x_std = fit_scaler(X_train_full)
                    y_mean, y_std = fit_scaler_1d(y_train_full)

                X_inj_s = apply_scaler(X_test_inj, x_mean, x_std)
                y_inj_s = apply_scaler_1d(y_test_inj, y_mean, y_std)
                inj_ldr = make_dataloader(X_inj_s, y_inj_s, cfg.train.batch_size, shuffle=False)

                model = make_model(model_type, len(feature_cols), cfg.data.window_size, cfg).to(device)
                state = torch.load(model_file, map_location=device)
                model.load_state_dict(state)
                y_injected, y_pred = predict(model, inj_ldr, device, y_scaler=(y_mean, y_std))
                del model

                errors_inj = np.abs(y_injected - y_pred)
                scores_inj = rolling_mad_scores(errors_inj, cfg.anomaly.mad_window,
                                                cfg.anomaly.epsilon)

                # Binary detection
                pred_labels = (scores_inj > cfg.anomaly.threshold).astype(int)
                precision = precision_score(inj_labels, pred_labels, zero_division=0)
                recall = recall_score(inj_labels, pred_labels, zero_division=0)
                f1 = f1_score(inj_labels, pred_labels, zero_division=0)

                try:
                    auroc = roc_auc_score(inj_labels, scores_inj)
                    auprc = average_precision_score(inj_labels, scores_inj)
                except ValueError:
                    auroc, auprc = float("nan"), float("nan")

                far = (pred_labels[inj_labels == 0].sum()) / max((inj_labels == 0).sum(), 1)

                row = {
                    "target": target_id, "model": model_type, "k": k,
                    "method": method_name, "seed": rep_seed,
                    "Precision": precision, "Recall": recall, "F1": f1,
                    "AUROC": auroc, "AUPRC": auprc, "FAR": far,
                }
                for anomaly_type in sorted(set(inj_types) - {"normal"}):
                    mask = (inj_types == anomaly_type) | (inj_types == f"{anomaly_type}")
                    if mask.sum() == 0:
                        continue
                    row[f"Recall_{anomaly_type}"] = recall_score(
                        inj_labels[mask], pred_labels[mask], zero_division=0
                    )
                all_injection_rows.append(row)

                # Save
                inj_out = inj_dir / model_type / f"k{k}" / method_name.split("_")[0]
                inj_out.mkdir(parents=True, exist_ok=True)
                np.savez(inj_out / "injection_results.npz",
                         y_injected=y_injected, y_pred=y_pred,
                         anomaly_scores=scores_inj, labels=inj_labels,
                         anomaly_types=inj_types)
                with open(inj_out / "injection_metrics.json", "w") as f:
                    json.dump(row, f, indent=2, default=str)


def _aggregate_results(out_root, building_ids, mdls, ks, sds):
    """Scan results directory and generate aggregate tables."""
    agg_dir = out_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Collect all individual results
    rows = []
    for bid in building_ids:
        for model in mdls:
            for k in ks:
                for seed in sds:
                    for method in ["M1", "M2", "M3"]:
                        if k == 0 and method == "M2":
                            continue
                        metrics_file = out_root / bid / model / f"k{k}" / f"seed{seed}" / method / "metrics.json"
                        if metrics_file.exists():
                            with open(metrics_file) as f:
                                m = json.load(f)
                            m["building"] = bid
                            rows.append(m)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(agg_dir / "table_few_shot_transfer.csv", index=False)
        print(f"[Phase2] Aggregate: {len(df)} rows → table_few_shot_transfer.csv")

    # Baselines
    baseline_rows = []
    for bid in building_ids:
        for bm in ["isolation_forest", "lof", "lstm_ae"]:
            mf = out_root / bid / "baselines" / bm / "metrics.json"
            if mf.exists():
                with open(mf) as f:
                    baseline_rows.append(json.load(f))
    if baseline_rows:
        pd.DataFrame(baseline_rows).to_csv(agg_dir / "table_baseline_comparison.csv", index=False)

    # Injection
    inj_rows = []
    for bid in building_ids:
        inj_root = out_root / bid / "anomaly_injection"
        for inj_file in inj_root.rglob("injection_metrics.json"):
            with open(inj_file) as f:
                inj_rows.append(json.load(f))
    if inj_rows:
        pd.DataFrame(inj_rows).to_csv(agg_dir / "table_anomaly_injection.csv", index=False)

    print(f"[Phase2] Aggregate tables saved to {agg_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: Full experiment matrix")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    parser.add_argument("--output", default="results/phase2")
    parser.add_argument("--n-buildings", type=int, default=12)
    parser.add_argument("--building-manifest", default=None)
    parser.add_argument("--k", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-injection", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 3 buildings, k=[7], seed=[42], LSTM only")
    args = parser.parse_args()

    if args.quick:
        run_phase2(
            config_path=args.config, data_path=args.data, output_dir=args.output,
            n_buildings=3, k_values=[7], seeds=[42], models=["lstm"],
            skip_baselines=True, skip_injection=True,
        )
    else:
        run_phase2(
            config_path=args.config, data_path=args.data, output_dir=args.output,
            n_buildings=args.n_buildings,
            building_manifest=args.building_manifest,
            k_values=args.k, seeds=args.seeds, models=args.models,
            skip_baselines=args.skip_baselines,
            skip_injection=args.skip_injection,
        )
