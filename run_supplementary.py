#!/usr/bin/env python
"""Run supplementary experiments not covered by Phase 1 or Phase 2.

Usage:
    python run_supplementary.py --scenario-a
    python run_supplementary.py --classical
    python run_supplementary.py --ablation-e2 --ablation-e4
    python run_supplementary.py --all
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from src.config import Config
from src.models import LSTMForecaster, DLinear
from src.train import (
    build_windows, fit_scaler, apply_scaler, make_dataloader,
    train_model, pretrain, fine_tune, predict, save_run,
    fit_scaler_1d, apply_scaler_1d,
)
from src.anomaly import rolling_mad_scores
from src.metrics import compute_all
from src.anomaly_injection import inject_anomalies

FEATURE_COLS = ["energy", "hour", "day_of_week", "is_weekend", "month"]
ENERGY_ONLY = ["energy"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ===================================================================
# Scenario A: Same-Domain Upper Bound (M5)
# ===================================================================

def run_scenario_a(cfg, data, output_dir="results/supplementary/scenario_a"):
    """Train each model on full 60% training data. No transfer."""
    print("\n" + "="*60)
    print("SCENARIO A: Same-Domain Upper Bound (M5)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    building_ids = sorted(data["building_id"].unique())[:5]
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    n_features = len(feature_cols)
    results = []

    for bid in building_ids:
        print(f"\n  Building: {bid}")
        bdata = data[data["building_id"] == bid].sort_values("timestamp")
        n = len(bdata)
        train_end = int(n * 0.6)
        val_end = int(n * (0.6 + 0.2))

        train_df = bdata.iloc[:train_end]
        val_df = bdata.iloc[train_end:val_end]
        test_df = bdata.iloc[val_end:]

        for model_name in ["lstm", "dlinear"]:
            X_tr, y_tr = build_windows(train_df, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
            X_val, y_val = build_windows(val_df, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
            X_test, y_test = build_windows(test_df, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)

            if len(X_tr) < 100:
                continue

            # Scale X + y
            x_mean, x_std = fit_scaler(X_tr)
            y_mean, y_std = fit_scaler_1d(y_tr)
            X_tr_s = apply_scaler(X_tr, x_mean, x_std)
            X_val_s = apply_scaler(X_val, x_mean, x_std)
            X_test_s = apply_scaler(X_test, x_mean, x_std)
            y_tr_s = apply_scaler_1d(y_tr, y_mean, y_std)
            y_val_s = apply_scaler_1d(y_val, y_mean, y_std)
            y_test_s = apply_scaler_1d(y_test, y_mean, y_std)

            train_ldr = make_dataloader(X_tr_s, y_tr_s, cfg.train.batch_size, shuffle=True)
            val_ldr = make_dataloader(X_val_s, y_val_s, cfg.train.batch_size, shuffle=False)
            test_ldr = make_dataloader(X_test_s, y_test_s, cfg.train.batch_size, shuffle=False)

            for seed in [42, 43, 44]:
                set_seed(seed)
                out_dir = Path(output_dir) / bid / model_name / f"seed{seed}"
                if (out_dir / "metrics.json").exists():
                    continue

                if model_name == "lstm":
                    model = LSTMForecaster(n_features, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout)
                else:
                    model = DLinear(cfg.data.window_size, n_features, 1)
                model = model.to(device)

                history = train_model(model, train_ldr, val_ldr, cfg, device, verbose=False)
                yt, yp = predict(model, test_ldr, device, y_scaler=(y_mean, y_std))
                err = np.abs(yt - yp)
                sc = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                metrics = compute_all(yt, yp, sc)
                metrics.update({"method": "M5_upper_bound", "model": model_name, "target": bid, "seed": seed})
                save_run(str(out_dir), cfg, metrics, predictions=(yt, yp), anomaly_scores=sc, history=history)
                results.append(metrics)
                del model
                print(f"    {model_name} seed={seed}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, sigma_err={metrics['sigma_err']:.2f}")

    if results:
        pd.DataFrame(results).to_csv(Path(output_dir) / "scenario_a_results.csv", index=False)
        print(f"\n  Saved {len(results)} runs to {output_dir}/scenario_a_results.csv")


# ===================================================================
# Classical Baselines: ARIMA, SARIMA, Naive
# ===================================================================

def run_classical_baselines(cfg, data, output_dir="results/supplementary/classical"):
    """ARIMA, SARIMA, Naive (persistence) baselines."""
    print("\n" + "="*60)
    print("CLASSICAL BASELINES: ARIMA, SARIMA, Naive")
    print("="*60)

    building_ids = sorted(data["building_id"].unique())[:5]
    results = []

    for bid in building_ids:
        print(f"\n  Building: {bid}")
        bdata = data[data["building_id"] == bid].sort_values("timestamp")
        n = len(bdata)
        val_end = int(n * 0.8)
        test_series = bdata.iloc[val_end:]["energy"].values
        if len(test_series) < 100:
            continue

        # Naive (persistence): y_hat[t+1] = y[t]
        err_naive = np.abs(test_series[1:] - test_series[:-1])
        sc_naive = rolling_mad_scores(err_naive, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
        metrics_naive = compute_all(test_series[1:], test_series[:-1], sc_naive)
        metrics_naive.update({"method": "Naive", "target": bid})
        results.append(metrics_naive)
        print(f"    Naive: MAE={metrics_naive['MAE']:.2f}, sigma_err={metrics_naive['sigma_err']:.2f}")

        # ARIMA — try with order auto-selection or fixed (1,0,1)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            train_series = bdata.iloc[:int(n * 0.6)]["energy"].values
            # Simple ARIMA(1,0,1) for speed
            model_arima = ARIMA(train_series, order=(1, 0, 1))
            fitted = model_arima.fit()
            forecasts = fitted.forecast(len(test_series))
            err_arima = np.abs(test_series - forecasts)
            sc_arima = rolling_mad_scores(err_arima, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
            metrics_arima = compute_all(test_series, forecasts, sc_arima)
            metrics_arima.update({"method": "ARIMA", "target": bid})
            results.append(metrics_arima)
            print(f"    ARIMA(1,0,1): MAE={metrics_arima['MAE']:.2f}, sigma_err={metrics_arima['sigma_err']:.2f}")
        except Exception as e:
            print(f"    ARIMA failed: {e}")

        # SARIMA(1,0,1)(1,0,1,24) for daily seasonality
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            train_series = bdata.iloc[:int(n * 0.6)]["energy"].values
            model_sarima = SARIMAX(train_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
            fitted_s = model_sarima.fit(disp=False)
            forecasts_s = fitted_s.forecast(len(test_series))
            err_sarima = np.abs(test_series - forecasts_s)
            sc_sarima = rolling_mad_scores(err_sarima, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
            metrics_sarima = compute_all(test_series, forecasts_s, sc_sarima)
            metrics_sarima.update({"method": "SARIMA", "target": bid})
            results.append(metrics_sarima)
            print(f"    SARIMA: MAE={metrics_sarima['MAE']:.2f}, sigma_err={metrics_sarima['sigma_err']:.2f}")
        except Exception as e:
            print(f"    SARIMA failed: {e}")

    if results:
        pd.DataFrame(results).to_csv(Path(output_dir) / "classical_baselines.csv", index=False)
        print(f"\n  Saved to {output_dir}/classical_baselines.csv")


# ===================================================================
# Ablation E2: Time Features
# ===================================================================

def run_ablation_e2(cfg, data, output_dir="results/supplementary/ablation_e2"):
    """Compare energy-only vs energy+time_features on M3, k=7."""
    print("\n" + "="*60)
    print("ABLATION E2: Time Features")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    building_ids = sorted(data["building_id"].unique())[:5]
    full_features = [c for c in FEATURE_COLS if c in data.columns]
    n_full = len(full_features)
    n_energy = 1  # energy only

    results = []

    for bid in building_ids:
        print(f"\n  Building: {bid}")
        source_ids = [b for b in building_ids if b != bid]
        bdata = data[data["building_id"] == bid].sort_values("timestamp")
        n_tgt = len(bdata)
        train_end = int(n_tgt * 0.6)
        val_end = int(n_tgt * (0.6 + 0.2))
        tgt_train = bdata.iloc[:train_end]
        tgt_val = bdata.iloc[train_end:val_end]
        tgt_test = bdata.iloc[val_end:]

        k_hours = 168  # 7 days
        tgt_few = tgt_train.iloc[:k_hours]

        for feat_set_label, feat_cols in [("full", full_features), ("energy_only", ENERGY_ONLY)]:
            n_feat = len(feat_cols)

            # Source loaders
            source_loaders = []
            for src_id in source_ids:
                src_data = data[data["building_id"] == src_id].sort_values("timestamp")
                n_src = len(src_data)
                src_train = src_data.iloc[:int(n_src * 0.6)]
                src_val = src_data.iloc[int(n_src * 0.6):int(n_src * 0.8)]
                X_tr, y_tr = build_windows(src_train, feat_cols, "energy", cfg.data.window_size, cfg.data.horizon)
                X_v, y_v = build_windows(src_val, feat_cols, "energy", cfg.data.window_size, cfg.data.horizon)
                if len(X_tr) < 10:
                    continue
                mx, sx = fit_scaler(X_tr)
                my, sy = fit_scaler_1d(y_tr)
                train_ldr = make_dataloader(apply_scaler(X_tr, mx, sx), apply_scaler_1d(y_tr, my, sy), cfg.train.batch_size, shuffle=True)
                val_ldr = make_dataloader(apply_scaler(X_v, mx, sx), apply_scaler_1d(y_v, my, sy), cfg.train.batch_size, shuffle=False)
                source_loaders.append((train_ldr, val_ldr))

            if not source_loaders:
                continue

            X_few, y_few = build_windows(tgt_few, feat_cols, "energy", cfg.data.window_size, cfg.data.horizon)
            X_val, y_val = build_windows(tgt_val, feat_cols, "energy", cfg.data.window_size, cfg.data.horizon)
            X_test, y_test = build_windows(tgt_test, feat_cols, "energy", cfg.data.window_size, cfg.data.horizon)

            if len(X_few) < 10:
                continue

            fx_m, fx_s = fit_scaler(X_few)
            fy_m, fy_s = fit_scaler_1d(y_few)
            X_few_s = apply_scaler(X_few, fx_m, fx_s)
            X_val_s = apply_scaler(X_val, fx_m, fx_s)
            X_test_s = apply_scaler(X_test, fx_m, fx_s)
            y_few_s = apply_scaler_1d(y_few, fy_m, fy_s)
            y_val_s = apply_scaler_1d(y_val, fy_m, fy_s)
            y_test_s = apply_scaler_1d(y_test, fy_m, fy_s)

            few_bs = max(4, min(cfg.train.batch_size, len(X_few_s) // 4))

            for seed in [42, 43, 44]:
                out_dir = Path(output_dir) / bid / feat_set_label / f"seed{seed}"
                if (out_dir / "metrics.json").exists():
                    continue

                set_seed(seed)
                model = LSTMForecaster(n_feat, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout).to(device)
                pretrain(model, source_loaders, cfg, device, verbose=False)
                ft_train = make_dataloader(X_few_s, y_few_s, few_bs, shuffle=True)
                ft_val = make_dataloader(X_val_s, y_val_s, few_bs, shuffle=False)
                fine_tune(model, ft_train, ft_val, cfg, device, verbose=False)
                test_ldr = make_dataloader(X_test_s, y_test_s, cfg.train.batch_size, shuffle=False)
                yt, yp = predict(model, test_ldr, device, y_scaler=(fy_m, fy_s))
                err = np.abs(yt - yp)
                sc = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                metrics = compute_all(yt, yp, sc)
                metrics.update({"method": "M3_pretrain_ft", "features": feat_set_label, "target": bid, "seed": seed})
                save_run(str(out_dir), cfg, metrics, predictions=(yt, yp), anomaly_scores=sc, history={})
                results.append(metrics)
                del model
                print(f"    {feat_set_label:12s} seed={seed}: MAE={metrics['MAE']:.2f}, sigma_err={metrics['sigma_err']:.2f}")

    if results:
        pd.DataFrame(results).to_csv(Path(output_dir) / "ablation_e2.csv", index=False)


# ===================================================================
# Ablation E4: Anomaly Scoring Method
# ===================================================================

def run_ablation_e4(cfg, data, phase2_results_dir="results/phase2",
                    output_dir="results/supplementary/ablation_e4"):
    """Compare static quantile threshold vs rolling MAD on Phase 2 predictions."""
    print("\n" + "="*60)
    print("ABLATION E4: Anomaly Scoring Method")
    print("="*60)

    results = []
    out_root = Path(phase2_results_dir)
    if not out_root.exists():
        print(f"  SKIP: Phase 2 results not found at {phase2_results_dir}")
        return

    for pred_file in out_root.rglob("predictions.npz"):
        try:
            pred_data = np.load(pred_file)
            y_true = pred_data["y_true"]
            y_pred = pred_data["y_pred"]
            errors = np.abs(y_true - y_pred)

            # Method 1: Rolling MAD (existing)
            scores_mad = rolling_mad_scores(errors, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
            labels_mad = (scores_mad > cfg.anomaly.threshold).astype(int)

            # Method 2: Static quantile (95th percentile)
            static_threshold = np.percentile(errors, 95)
            scores_static = errors / (static_threshold + cfg.anomaly.epsilon)
            labels_static = (errors > static_threshold).astype(int)

            # Count "anomaly" rate on clean data
            far_mad = labels_mad.mean()
            far_static = labels_static.mean()

            # Variability
            cv_mad = float(np.std(scores_mad) / (np.mean(scores_mad) + 1e-8))
            cv_static = float(np.std(scores_static) / (np.mean(scores_static) + 1e-8))

            # Parse path for metadata
            path_str = str(pred_file.parent)
            parts = Path(path_str).parts
            m = {"path": path_str}
            for part in parts:
                if part.startswith("k") and len(part) <= 3:
                    m["k"] = int(part[1:])
                elif part in ("M2", "M3"):
                    m["method"] = "M2_target_only" if part == "M2" else "M3_pretrain_ft"

            m.update({
                "sigma_err": float(np.std(errors)),
                "mad_sigma_score": float(np.std(scores_mad)),
                "static_sigma_score": float(np.std(scores_static)),
                "mad_far": far_mad,
                "static_far": far_static,
                "mad_cv": cv_mad,
                "static_cv": cv_static,
            })
            results.append(m)
        except Exception:
            pass

    if results:
        pd.DataFrame(results).to_csv(Path(output_dir) / "ablation_e4.csv", index=False)
        print(f"  Saved {len(results)} entries to {output_dir}/ablation_e4.csv")


# ===================================================================
# Ablation E5: Freeze + Fine-tune (M4) — LSTM only
# ===================================================================

def freeze_backbone_ft(model, train_loader, val_loader, config, device):
    """Fine-tune only the FC head, freeze LSTM backbone."""
    # Freeze all LSTM parameters
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=config.train.finetune_lr,
        weight_decay=config.train.weight_decay,
    )
    criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(config.train.finetune_epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb).squeeze(-1)
                val_loss += criterion(pred, yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.train.early_stopping_patience:
            break

    model.load_state_dict(best_weights)
    # Unfreeze
    for p in model.parameters():
        p.requires_grad = True
    return {"best_val_loss": best_val_loss}


def run_ablation_e5(cfg, data, output_dir="results/supplementary/ablation_e5"):
    """Compare M3 (full FT) vs M4 (freeze+FT) for LSTM."""
    print("\n" + "="*60)
    print("ABLATION E5: Freeze + Fine-tune (M4)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    building_ids = sorted(data["building_id"].unique())[:5]
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    n_features = len(feature_cols)
    results = []

    for bid in building_ids:
        print(f"\n  Building: {bid}")
        source_ids = [b for b in building_ids if b != bid]
        bdata = data[data["building_id"] == bid].sort_values("timestamp")
        n_tgt = len(bdata)
        train_end = int(n_tgt * 0.6)
        val_end = int(n_tgt * (0.6 + 0.2))
        tgt_val = bdata.iloc[train_end:val_end]
        tgt_test = bdata.iloc[val_end:]
        tgt_few = bdata.iloc[:train_end].iloc[:168]  # 7 days

        # Source loaders
        source_loaders = []
        for src_id in source_ids:
            src_data = data[data["building_id"] == src_id].sort_values("timestamp")
            n_src = len(src_data)
            src_train = src_data.iloc[:int(n_src * 0.6)]
            src_val = src_data.iloc[int(n_src * 0.6):int(n_src * 0.8)]
            X_tr, y_tr = build_windows(src_train, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
            X_v, y_v = build_windows(src_val, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
            if len(X_tr) < 10:
                continue
            mx, sx = fit_scaler(X_tr)
            my, sy = fit_scaler_1d(y_tr)
            train_ldr = make_dataloader(apply_scaler(X_tr, mx, sx), apply_scaler_1d(y_tr, my, sy), cfg.train.batch_size, shuffle=True)
            val_ldr = make_dataloader(apply_scaler(X_v, mx, sx), apply_scaler_1d(y_v, my, sy), cfg.train.batch_size, shuffle=False)
            source_loaders.append((train_ldr, val_ldr))

        if not source_loaders:
            continue

        X_few, y_few = build_windows(tgt_few, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
        X_val_t, y_val_t = build_windows(tgt_val, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
        X_test_t, y_test_t = build_windows(tgt_test, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)

        if len(X_few) < 10:
            continue

        fx_m, fx_s = fit_scaler(X_few)
        fy_m, fy_s = fit_scaler_1d(y_few)
        X_few_s = apply_scaler(X_few, fx_m, fx_s)
        X_val_s = apply_scaler(X_val_t, fx_m, fx_s)
        X_test_s = apply_scaler(X_test_t, fx_m, fx_s)
        y_few_s = apply_scaler_1d(y_few, fy_m, fy_s)
        y_val_s = apply_scaler_1d(y_val_t, fy_m, fy_s)
        y_test_s = apply_scaler_1d(y_test_t, fy_m, fy_s)

        few_bs = max(4, min(cfg.train.batch_size, len(X_few_s) // 4))

        for seed in [42, 43, 44]:
            # M3: full FT (same as before)
            m3_dir = Path(output_dir) / bid / f"M3_full" / f"seed{seed}"
            if not (m3_dir / "metrics.json").exists():
                set_seed(seed)
                model = LSTMForecaster(n_features, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout).to(device)
                pretrain(model, source_loaders, cfg, device, verbose=False)
                ft_train = make_dataloader(X_few_s, y_few_s, few_bs, shuffle=True)
                ft_val = make_dataloader(X_val_s, y_val_s, few_bs, shuffle=False)
                fine_tune(model, ft_train, ft_val, cfg, device, verbose=False)
                test_ldr = make_dataloader(X_test_s, y_test_s, cfg.train.batch_size, shuffle=False)
                yt, yp = predict(model, test_ldr, device, y_scaler=(fy_m, fy_s))
                err = np.abs(yt - yp)
                sc = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                metrics = compute_all(yt, yp, sc)
                metrics.update({"method": "M3_full_ft", "target": bid, "seed": seed})
                save_run(str(m3_dir), cfg, metrics, predictions=(yt, yp), anomaly_scores=sc, history={})
                results.append(metrics)
                del model
                print(f"    M3 full FT seed={seed}: MAE={metrics['MAE']:.2f}, sigma_err={metrics['sigma_err']:.2f}")

            # M4: freeze+FT
            m4_dir = Path(output_dir) / bid / f"M4_freeze" / f"seed{seed}"
            if not (m4_dir / "metrics.json").exists():
                set_seed(seed)
                model = LSTMForecaster(n_features, cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.dropout).to(device)
                pretrain(model, source_loaders, cfg, device, verbose=False)
                ft_train = make_dataloader(X_few_s, y_few_s, few_bs, shuffle=True)
                ft_val = make_dataloader(X_val_s, y_val_s, few_bs, shuffle=False)
                freeze_backbone_ft(model, ft_train, ft_val, cfg, device)
                test_ldr = make_dataloader(X_test_s, y_test_s, cfg.train.batch_size, shuffle=False)
                yt, yp = predict(model, test_ldr, device, y_scaler=(fy_m, fy_s))
                err = np.abs(yt - yp)
                sc = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                metrics = compute_all(yt, yp, sc)
                metrics.update({"method": "M4_freeze_ft", "target": bid, "seed": seed})
                save_run(str(m4_dir), cfg, metrics, predictions=(yt, yp), anomaly_scores=sc, history={})
                results.append(metrics)
                del model
                print(f"    M4 freeze FT seed={seed}: MAE={metrics['MAE']:.2f}, sigma_err={metrics['sigma_err']:.2f}")

    if results:
        pd.DataFrame(results).to_csv(Path(output_dir) / "ablation_e5.csv", index=False)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Supplementary experiments")
    parser.add_argument("--scenario-a", action="store_true", help="Same-domain upper bound (M5)")
    parser.add_argument("--classical", action="store_true", help="ARIMA/SARIMA/Naive baselines")
    parser.add_argument("--ablation-e2", action="store_true", help="Time features ablation")
    parser.add_argument("--ablation-e4", action="store_true", help="Anomaly scoring method ablation")
    parser.add_argument("--ablation-e5", action="store_true", help="Freeze+FT ablation (M4)")
    parser.add_argument("--all", action="store_true", help="Run all supplementary experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    parser.add_argument("--buildings", type=int, default=5)
    args = parser.parse_args()

    if not any([args.scenario_a, args.classical, args.ablation_e2, args.ablation_e4, args.ablation_e5, args.all]):
        print("Nothing selected. Use --scenario-a, --classical, --ablation-e2/e4/e5, or --all")
        return

    cfg = Config.from_yaml(args.config)
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    building_ids = sorted(data["building_id"].unique())
    if len(building_ids) > args.buildings:
        building_ids = building_ids[:args.buildings]

    run_all = args.all

    if run_all or args.scenario_a:
        run_scenario_a(cfg, data)
    if run_all or args.classical:
        run_classical_baselines(cfg, data)
    if run_all or args.ablation_e2:
        run_ablation_e2(cfg, data)
    if run_all or args.ablation_e4:
        run_ablation_e4(cfg, data)
    if run_all or args.ablation_e5:
        run_ablation_e5(cfg, data)


if __name__ == "__main__":
    main()
