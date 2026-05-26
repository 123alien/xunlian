#!/usr/bin/env python
"""Pilot persistence-anchored and similarity-aware SRFT variants."""
import argparse
import json
import sys
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anomaly import rolling_mad_scores  # noqa: E402
from src.config import Config  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.phase2 import FEATURE_COLS, make_model, set_seed, _prepare_source_loaders  # noqa: E402
from src.train import (  # noqa: E402
    apply_scaler,
    apply_scaler_1d,
    build_windows,
    fit_scaler,
    fit_scaler_1d,
    make_dataloader,
    fine_tune,
    pretrain,
    save_run,
)


ENERGY_IDX = 0


def residual_target(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - X[:, -1, ENERGY_IDX]


def residual_predict_raw(model, loader, device, y_scaler):
    model.eval()
    y_true_list, pred_list = [], []
    mean, std = y_scaler
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            residual_s = model(X_batch).squeeze(-1).cpu().numpy()
            X_np = X_batch.cpu().numpy()
            last_energy_s = X_np[:, -1, ENERGY_IDX]
            pred_s = last_energy_s + residual_s
            y_true = y_batch.numpy() * std + mean
            y_pred = pred_s * std + mean
            y_true_list.append(y_true)
            pred_list.append(y_pred)
    return np.concatenate(y_true_list), np.concatenate(pred_list)


def split_building(data, bid):
    g = data[data["building_id"] == bid].sort_values("timestamp")
    n = len(g)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return g.iloc[:train_end], g.iloc[train_end:val_end], g.iloc[val_end:]


def make_residual_arrays(data, bid, k, feature_cols, cfg):
    train_full, val, test = split_building(data, bid)
    train_few = train_full.iloc[: k * 24]
    X_few_raw, y_few_raw = build_windows(train_few, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    X_val_raw, y_val_raw = build_windows(val, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    X_test_raw, y_test_raw = build_windows(test, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    if len(X_few_raw) < 10 or len(X_val_raw) == 0 or len(X_test_raw) == 0:
        return None
    x_mean, x_std = fit_scaler(X_few_raw)
    y_mean, y_std = fit_scaler_1d(y_few_raw)
    X_few_s = apply_scaler(X_few_raw, x_mean, x_std)
    X_val_s = apply_scaler(X_val_raw, x_mean, x_std)
    X_test_s = apply_scaler(X_test_raw, x_mean, x_std)
    y_few_s = apply_scaler_1d(y_few_raw, y_mean, y_std)
    y_val_s = apply_scaler_1d(y_val_raw, y_mean, y_std)
    y_test_s = apply_scaler_1d(y_test_raw, y_mean, y_std)
    return {
        "X_few": X_few_s,
        "y_few_res": residual_target(X_few_s, y_few_s),
        "X_val": X_val_s,
        "y_val_res": residual_target(X_val_s, y_val_s),
        "y_val": y_val_s,
        "X_test": X_test_s,
        "y_test": y_test_s,
        "y_scaler": (y_mean, y_std),
    }


def signature_from_train(data, bid, k_days=None):
    train, _, _ = split_building(data, bid)
    if k_days is not None:
        train = train.iloc[: k_days * 24]
    e = train["energy"].to_numpy(dtype=float)
    if len(e) < 2:
        return None
    by_hour = train.groupby(train["timestamp"].dt.hour)["energy"].mean().reindex(range(24)).fillna(0).to_numpy()
    diff = np.diff(e)
    return np.concatenate([
        np.array([
            np.mean(e),
            np.std(e),
            np.mean(np.isclose(e, 0.0)),
            np.percentile(e, 95) - np.percentile(e, 5),
            np.mean(np.abs(diff)) if len(diff) else 0.0,
        ]),
        by_hour / max(np.mean(e), 1e-6),
    ])


def source_weights(data, source_ids, target_id, k, tau=1.0, top_k=None):
    target_sig = signature_from_train(data, target_id, k_days=k)
    rows = []
    sigs = []
    for sid in source_ids:
        sig = signature_from_train(data, sid, k_days=None)
        if sig is None:
            continue
        rows.append(sid)
        sigs.append(sig)
    if target_sig is None or not sigs:
        return {sid: 1 / max(len(source_ids), 1) for sid in source_ids}
    mat = np.vstack(sigs)
    ref = target_sig.copy()
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd < 1e-8] = 1.0
    d = np.linalg.norm((mat - mu) / sd - (ref - mu) / sd, axis=1)
    if top_k is not None and top_k < len(d):
        keep = np.argsort(d)[:top_k]
        mask = np.zeros_like(d, dtype=bool)
        mask[keep] = True
        d = np.where(mask, d, np.inf)
    score = -d / max(tau, 1e-6)
    score = score - np.nanmax(score[np.isfinite(score)])
    w = np.exp(score)
    w[~np.isfinite(w)] = 0.0
    w = w / max(w.sum(), 1e-12)
    return {sid: float(wi) for sid, wi in zip(rows, w)}


def residual_source_loader(data, bid, feature_cols, cfg, batch_size):
    train, val, _ = split_building(data, bid)
    X_tr_raw, y_tr_raw = build_windows(train, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    X_v_raw, y_v_raw = build_windows(val, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    if len(X_tr_raw) < 10 or len(X_v_raw) == 0:
        return None
    x_mean, x_std = fit_scaler(X_tr_raw)
    y_mean, y_std = fit_scaler_1d(y_tr_raw)
    X_tr = apply_scaler(X_tr_raw, x_mean, x_std)
    X_v = apply_scaler(X_v_raw, x_mean, x_std)
    y_tr = apply_scaler_1d(y_tr_raw, y_mean, y_std)
    y_v = apply_scaler_1d(y_v_raw, y_mean, y_std)
    return (
        make_dataloader(X_tr, residual_target(X_tr, y_tr), batch_size, shuffle=True),
        make_dataloader(X_v, residual_target(X_v, y_v), batch_size, shuffle=False),
    )


def residual_replay_finetune(model, target_loader, val_loader, source_loaders, source_probs, cfg, device, lambda_source):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.finetune_lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    patience = 0
    history = {"train_loss": [], "target_loss": [], "source_loss": [], "val_loss": []}
    loaders = [p[1][0] for p in source_loaders]
    cycles = [cycle(x) for x in loaders]
    probs = np.array([source_probs.get(p[0], 0.0) for p in source_loaders], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(cycles)) / len(cycles)

    for epoch in range(cfg.train.finetune_epochs):
        model.train()
        total = target_total = source_total = n = 0.0
        for X_t, r_t in target_loader:
            X_t, r_t = X_t.to(device), r_t.to(device)
            idx = int(np.random.choice(len(cycles), p=probs))
            X_s, r_s = next(cycles[idx])
            X_s, r_s = X_s.to(device), r_s.to(device)
            optimizer.zero_grad()
            pred_t = model(X_t).squeeze(-1)
            pred_s = model(X_s).squeeze(-1)
            lt = criterion(pred_t, r_t)
            ls = criterion(pred_s, r_s)
            loss = lt + lambda_source * ls
            loss.backward()
            optimizer.step()
            bs = X_t.size(0)
            total += float(loss.item()) * bs
            target_total += float(lt.item()) * bs
            source_total += float(ls.item()) * bs
            n += bs

        model.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for X_v, r_v in val_loader:
                X_v, r_v = X_v.to(device), r_v.to(device)
                loss = criterion(model(X_v).squeeze(-1), r_v)
                vloss += float(loss.item()) * X_v.size(0)
                vn += X_v.size(0)
        vloss /= max(vn, 1)
        history["train_loss"].append(total / max(n, 1))
        history["target_loss"].append(target_total / max(n, 1))
        history["source_loss"].append(source_total / max(n, 1))
        history["val_loss"].append(vloss)
        if vloss < best_loss:
            best_loss = vloss
            best_weights = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= cfg.train.early_stopping_patience:
            break
    model.load_state_dict(best_weights)
    history["best_val_loss"] = best_loss
    history["stopped_epoch"] = epoch + 1
    return history


def save_metrics(model, loader, arrays, cfg, device, run_dir, meta, history, manifest):
    y_true, y_pred = residual_predict_raw(model, loader, device, arrays["y_scaler"])
    err = np.abs(y_true - y_pred)
    scores = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
    metrics = compute_all(y_true, y_pred, scores)
    metrics.update(meta)
    save_run(str(run_dir), cfg, metrics, predictions=(y_true, y_pred), anomaly_scores=scores,
             history=history, model_state=model.state_dict(), manifest=manifest)


def aggregate(out_root):
    rows = []
    for path in out_root.glob("*/**/metrics.json"):
        row = json.loads(path.read_text())
        if str(row.get("method", "")).startswith("PA_") or str(row.get("method", "")).startswith("Residual") or str(row.get("method", "")).startswith("Sim"):
            rows.append(row)
    df = pd.DataFrame(rows)
    agg = out_root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    df.to_csv(agg / "table_pa_srft_pilot.csv", index=False)
    if not df.empty:
        print(df.groupby(["method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index().to_string(index=False))
    print(f"[PA-SRFT] rows={len(df)} saved={agg / 'table_pa_srft_pilot.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly_expanded.csv")
    ap.add_argument("--phase2-dir", default="results/phase2_paper/expanded_fast/neural")
    ap.add_argument("--output-dir", default="results/pa_srft_pilot")
    ap.add_argument("--building-manifest", required=True)
    ap.add_argument("--n-buildings", type=int, default=8)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7, 14, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--lambda-source", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--top-source-k", type=int, default=5)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    manifest = pd.read_csv(args.building_manifest)
    id_col = "building_id" if "building_id" in manifest.columns else "target"
    building_ids = manifest[id_col].dropna().astype(str).tolist()[: args.n_buildings]
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    out_root = Path(args.output_dir)
    phase2 = Path(args.phase2_dir)

    for target_id in building_ids:
        source_ids = [b for b in building_ids if b != target_id]
        for k in args.k:
            arrays = make_residual_arrays(data, target_id, k, feature_cols, cfg)
            if arrays is None:
                continue
            few_bs = max(4, min(cfg.train.batch_size, len(arrays["X_few"]) // 4))
            target_loader = make_dataloader(arrays["X_few"], arrays["y_few_res"], few_bs, shuffle=True)
            val_loader = make_dataloader(arrays["X_val"], arrays["y_val_res"], few_bs, shuffle=False)
            test_loader = make_dataloader(arrays["X_test"], arrays["y_test"], cfg.train.batch_size, shuffle=False)
            src_loaders = []
            for sid in source_ids:
                pair = residual_source_loader(data, sid, feature_cols, cfg, cfg.train.batch_size)
                if pair is not None:
                    src_loaders.append((sid, pair))
            sim_weights = source_weights(data, [x[0] for x in src_loaders], target_id, k, tau=args.tau, top_k=args.top_source_k)
            uniform_weights = {sid: 1 / max(len(src_loaders), 1) for sid, _ in src_loaders}

            for seed in args.seeds:
                residual_pretrain_dir = out_root / target_id / "dlinear" / "residual_pretrain" / f"seed{seed}"
                residual_pretrain_ckpt = residual_pretrain_dir / "model.pt"
                if residual_pretrain_ckpt.exists():
                    residual_state = torch.load(residual_pretrain_ckpt, map_location=device)
                    pretrain_hist = json.loads((residual_pretrain_dir / "history.json").read_text()) if (residual_pretrain_dir / "history.json").exists() else {}
                else:
                    set_seed(seed)
                    model_pre = make_model("dlinear", len(feature_cols), cfg.data.window_size, cfg).to(device)
                    loader_pairs = [pair for _, pair in src_loaders]
                    pretrain_hist = pretrain(model_pre, loader_pairs, cfg, device, verbose=False)
                    residual_pretrain_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model_pre.state_dict(), residual_pretrain_ckpt)
                    (residual_pretrain_dir / "history.json").write_text(json.dumps(pretrain_hist, indent=2), encoding="utf-8")
                    residual_state = deepcopy(model_pre.state_dict())
                    del model_pre

                for method, weights, use_replay in [
                    ("Residual_M3", {}, False),
                    ("Residual_SRFT", uniform_weights, True),
                    ("Sim_SRFT", sim_weights, True),
                    ("PA_SRFT", sim_weights, True),
                ]:
                    run_dir = out_root / target_id / "dlinear" / f"k{k}" / f"seed{seed}" / method
                    if (run_dir / "metrics.json").exists():
                        continue
                    set_seed(seed)
                    model = make_model("dlinear", len(feature_cols), cfg.data.window_size, cfg).to(device)
                    model.load_state_dict(residual_state)
                    if use_replay and src_loaders:
                        hist = residual_replay_finetune(model, target_loader, val_loader, src_loaders, weights, cfg, device, args.lambda_source)
                    else:
                        hist = fine_tune(model, target_loader, val_loader, cfg, device, verbose=False)
                    hist = {"residual_pretrain": pretrain_hist, "residual_finetune": hist}
                    meta = {
                        "method": method,
                        "model": "dlinear",
                        "target": target_id,
                        "k": int(k),
                        "seed": int(seed),
                        "lambda_source": float(args.lambda_source) if use_replay else 0.0,
                    }
                    save_metrics(
                        model, test_loader, arrays, cfg, device, run_dir, meta, hist,
                        {
                            "method": method,
                            "base_checkpoint": str(residual_pretrain_ckpt),
                            "residual_anchor": "last_observed_energy",
                            "similarity_weights": weights if use_replay else {},
                            "tau": args.tau,
                            "top_source_k": args.top_source_k,
                        },
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"[PA-SRFT] {target_id} k{k} seed{seed} {method}", flush=True)
    aggregate(out_root)


if __name__ == "__main__":
    main()
