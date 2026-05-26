#!/usr/bin/env python
"""M3R: source-replay fine-tuning for cold-start forecasting.

Main-line purpose:
Improve the forecasting-focused paper by testing whether pretrained neural
models can better compete with strong source+target classical baselines when
fine-tuning keeps a small source replay loss.

M3R starts from the existing source-only M1 checkpoint at k=0, then fine-tunes
on target few-shot data while replaying source-building batches:

    loss = target_loss + lambda_source * source_loss
"""
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

sys.path.insert(0, str(Path(__file__).parent))

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
    predict,
    save_run,
)


def replay_fine_tune(model, target_loader, val_loader, source_loaders,
                     cfg: Config, device, lambda_source: float,
                     verbose: bool = False) -> dict:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.finetune_lr,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_weights = deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "target_loss": [], "source_loss": [], "val_loss": []}

    source_train_loaders = [pair[0] for pair in source_loaders]
    source_cycles = [cycle(loader) for loader in source_train_loaders]
    source_idx = 0

    for epoch in range(cfg.train.finetune_epochs):
        model.train()
        total_loss = 0.0
        total_target = 0.0
        total_source = 0.0
        n_samples = 0

        for X_t, y_t in target_loader:
            X_t = X_t.to(device)
            y_t = y_t.to(device)

            loader_pos = source_idx % len(source_cycles)
            X_s, y_s = next(source_cycles[loader_pos])
            source_idx += 1
            X_s = X_s.to(device)
            y_s = y_s.to(device)

            optimizer.zero_grad()
            pred_t = model(X_t).squeeze(-1)
            pred_s = model(X_s).squeeze(-1)
            target_loss = criterion(pred_t, y_t)
            source_loss = criterion(pred_s, y_s)
            loss = target_loss + lambda_source * source_loss
            loss.backward()
            optimizer.step()

            bs = X_t.size(0)
            total_loss += float(loss.item()) * bs
            total_target += float(target_loss.item()) * bs
            total_source += float(source_loss.item()) * bs
            n_samples += bs

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v = X_v.to(device)
                y_v = y_v.to(device)
                pred_v = model(X_v).squeeze(-1)
                loss_v = criterion(pred_v, y_v)
                val_loss += float(loss_v.item()) * X_v.size(0)
                val_n += X_v.size(0)
        val_loss /= max(val_n, 1)

        history["train_loss"].append(total_loss / max(n_samples, 1))
        history["target_loss"].append(total_target / max(n_samples, 1))
        history["source_loss"].append(total_source / max(n_samples, 1))
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.train.early_stopping_patience:
            if verbose:
                print(f"[M3R] early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_weights)
    history["best_val_loss"] = best_val_loss
    history["stopped_epoch"] = epoch + 1
    history["lambda_source"] = lambda_source
    return history


def prepare_target_arrays(data, target_id: str, k: int, feature_cols: list[str], cfg: Config):
    tgt = data[data["building_id"] == target_id].sort_values("timestamp")
    n = len(tgt)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_full = tgt.iloc[:train_end]
    val = tgt.iloc[train_end:val_end]
    test = tgt.iloc[val_end:]

    train_few = train_full.iloc[:k * 24]
    X_few, y_few = build_windows(train_few, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    X_val, y_val = build_windows(val, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    X_test, y_test = build_windows(test, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    if len(X_few) < 10 or len(X_val) == 0 or len(X_test) == 0:
        return None

    x_mean, x_std = fit_scaler(X_few)
    y_mean, y_std = fit_scaler_1d(y_few)
    return {
        "X_few": apply_scaler(X_few, x_mean, x_std),
        "y_few_s": apply_scaler_1d(y_few, y_mean, y_std),
        "X_val": apply_scaler(X_val, x_mean, x_std),
        "y_val_s": apply_scaler_1d(y_val, y_mean, y_std),
        "X_test": apply_scaler(X_test, x_mean, x_std),
        "y_test_s": apply_scaler_1d(y_test, y_mean, y_std),
        "y_scaler": (y_mean, y_std),
    }


def aggregate_results(out_root: Path):
    rows = []
    for metrics_path in out_root.glob("*/**/metrics.json"):
        try:
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("method") == "M3R_replay_ft":
            rows.append(row)
    df = pd.DataFrame(rows)
    agg = out_root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    detail = agg / "table_m3r_replay.csv"
    df.to_csv(detail, index=False)
    if not df.empty:
        summary = df.groupby(["model", "k", "lambda_source"])[
            ["MAE", "RMSE", "sMAPE", "sigma_err", "sigma_score"]
        ].mean().reset_index()
        summary.to_csv(agg / "table_m3r_replay_summary.csv", index=False)
        print(summary.to_string(index=False))
    print(f"[M3R] saved {detail} rows={len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    ap.add_argument("--phase2-dir", default="results/phase2_paper")
    ap.add_argument("--output-dir", default="results/phase2_paper")
    ap.add_argument("--n-buildings", type=int, default=12)
    ap.add_argument("--building-manifest", default=None)
    ap.add_argument("--models", nargs="+", default=["lstm", "dlinear"])
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--lambda-source", type=float, nargs="+", default=[0.1])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    if args.building_manifest:
        manifest = pd.read_csv(args.building_manifest)
        building_ids = manifest["building_id"].dropna().astype(str).tolist()[:args.n_buildings]
    else:
        building_ids = sorted(data["building_id"].unique())[:args.n_buildings]
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    phase2 = Path(args.phase2_dir)
    out_root = Path(args.output_dir)

    for target_id in building_ids:
        source_ids = [b for b in building_ids if b != target_id]
        source_loaders, _ = _prepare_source_loaders(data, source_ids, feature_cols, cfg, device)
        if not source_loaders:
            continue
        for model_type in args.models:
            for k in args.k:
                arrays = prepare_target_arrays(data, target_id, k, feature_cols, cfg)
                if arrays is None:
                    continue
                few_bs = max(4, min(cfg.train.batch_size, len(arrays["X_few"]) // 4))
                target_loader = make_dataloader(arrays["X_few"], arrays["y_few_s"], few_bs, shuffle=True)
                val_loader = make_dataloader(arrays["X_val"], arrays["y_val_s"], few_bs, shuffle=False)
                test_loader = make_dataloader(arrays["X_test"], arrays["y_test_s"], cfg.train.batch_size, shuffle=False)

                for seed in args.seeds:
                    pretrain_ckpt = phase2 / target_id / model_type / "k0" / f"seed{seed}" / "M1" / "model.pt"
                    if not pretrain_ckpt.exists():
                        print(f"[M3R] missing source checkpoint {pretrain_ckpt}")
                        continue
                    for lambda_source in args.lambda_source:
                        run_dir = (
                            out_root / target_id / model_type / f"k{k}" / f"seed{seed}"
                            / f"M3R_lambda{lambda_source:g}"
                        )
                        if (run_dir / "metrics.json").exists():
                            continue
                        set_seed(seed)
                        model = make_model(model_type, len(feature_cols), cfg.data.window_size, cfg).to(device)
                        model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))
                        hist = replay_fine_tune(
                            model, target_loader, val_loader, source_loaders,
                            cfg, device, lambda_source=lambda_source, verbose=False
                        )
                        y_true, y_pred = predict(model, test_loader, device, y_scaler=arrays["y_scaler"])
                        err = np.abs(y_true - y_pred)
                        scores = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
                        metrics = compute_all(y_true, y_pred, scores)
                        metrics.update({
                            "method": "M3R_replay_ft",
                            "model": model_type,
                            "target": target_id,
                            "k": int(k),
                            "seed": int(seed),
                            "lambda_source": float(lambda_source),
                        })
                        save_run(
                            str(run_dir), cfg, metrics,
                            predictions=(y_true, y_pred),
                            anomaly_scores=scores,
                            history=hist,
                            model_state=model.state_dict(),
                            manifest={
                                "method": "M3R_replay_ft",
                                "base_checkpoint": str(pretrain_ckpt),
                                "lambda_source": lambda_source,
                                "paper_mainline": "cold_start_building_energy_forecasting",
                            },
                        )
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(f"[M3R] {target_id} {model_type} k{k} seed{seed} lambda={lambda_source}", flush=True)

    aggregate_results(out_root)


if __name__ == "__main__":
    main()
