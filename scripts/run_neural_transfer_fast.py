#!/usr/bin/env python
"""Fast neural transfer matrix with reusable source-pretrained checkpoints."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anomaly import rolling_mad_scores  # noqa: E402
from src.config import Config  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.phase2 import FEATURE_COLS, make_model, set_seed, _prepare_source_loaders  # noqa: E402
from src.train import (  # noqa: E402
    apply_scaler,
    apply_scaler_1d,
    build_windows,
    fine_tune,
    fit_scaler,
    fit_scaler_1d,
    make_dataloader,
    predict,
    pretrain,
    save_run,
    train_model,
)


def target_split(data: pd.DataFrame, target_id: str):
    tgt = data[data["building_id"] == target_id].sort_values("timestamp")
    n = len(tgt)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return tgt.iloc[:train_end], tgt.iloc[train_end:val_end], tgt.iloc[val_end:]


def eval_and_save(model, test_loader, y_scaler, cfg, meta, run_dir: Path, history=None, manifest=None):
    y_true, y_pred = predict(model, test_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"), y_scaler)
    err = np.abs(y_true - y_pred)
    scores = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
    metrics = compute_all(y_true, y_pred, scores)
    metrics.update(meta)
    save_run(
        str(run_dir),
        cfg,
        metrics,
        predictions=(y_true, y_pred),
        anomaly_scores=scores,
        history=history or {},
        model_state=model.state_dict(),
        manifest=manifest or meta,
    )


def aggregate(out_root: Path, building_ids: list[str], models: list[str], ks: list[int], seeds: list[int]):
    rows = []
    for bid in building_ids:
        for model in models:
            for k in ks:
                for seed in seeds:
                    for method_dir in ["M1", "M2", "M3"]:
                        if k > 0 and method_dir == "M1":
                            continue
                        if k == 0 and method_dir == "M2":
                            continue
                        mf = out_root / bid / model / f"k{k}" / f"seed{seed}" / method_dir / "metrics.json"
                        if not mf.exists():
                            continue
                        row = json.loads(mf.read_text())
                        row["building"] = bid
                        rows.append(row)
    agg = out_root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(agg / "table_few_shot_transfer.csv", index=False)
    df.to_csv(agg / "table_few_shot_transfer_all_models.csv", index=False)
    print(f"[fast-neural] aggregate rows={len(df)} saved={agg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly_expanded.csv")
    ap.add_argument("--output-dir", default="results/phase2_paper/expanded/neural_fast")
    ap.add_argument("--n-buildings", type=int, default=24)
    ap.add_argument("--building-manifest", default="results/phase2_paper/expanded/building_manifest_24.csv")
    ap.add_argument("--models", nargs="+", default=["lstm", "dlinear", "patchtst"])
    ap.add_argument("--k", type=int, nargs="+", default=[0, 3, 7, 14, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    manifest = pd.read_csv(args.building_manifest)
    building_ids = manifest["building_id"].dropna().astype(str).tolist()[:args.n_buildings]
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    out_root = Path(args.output_dir)
    ks = sorted(set(args.k))

    print(f"[fast-neural] device={device} buildings={len(building_ids)} models={args.models} k={ks}", flush=True)
    for bidx, target_id in enumerate(building_ids, 1):
        train_full, val, test = target_split(data, target_id)
        X_val_raw, y_val_raw = build_windows(val, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
        X_test_raw, y_test_raw = build_windows(test, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
        if len(X_test_raw) == 0 or len(X_val_raw) == 0:
            continue
        source_ids = [b for b in building_ids if b != target_id]
        source_loaders, _ = _prepare_source_loaders(data, source_ids, feature_cols, cfg, device)
        if not source_loaders:
            continue

        for model_type in args.models:
            print(f"[fast-neural] target {bidx}/{len(building_ids)} {target_id} model={model_type}", flush=True)
            for seed in args.seeds:
                pretrain_dir = out_root / target_id / model_type / "pretrain" / f"seed{seed}"
                pretrained_state_path = pretrain_dir / "model.pt"
                if pretrained_state_path.exists():
                    pretrained_state = torch.load(pretrained_state_path, map_location=device)
                    hist_pt = json.loads((pretrain_dir / "history.json").read_text()) if (pretrain_dir / "history.json").exists() else {}
                else:
                    set_seed(seed)
                    base_model = make_model(model_type, len(feature_cols), cfg.data.window_size, cfg).to(device)
                    hist_pt = pretrain(base_model, source_loaders, cfg, device, verbose=False)
                    pretrain_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(base_model.state_dict(), pretrained_state_path)
                    (pretrain_dir / "history.json").write_text(json.dumps(hist_pt, indent=2), encoding="utf-8")
                    pretrained_state = deepcopy(base_model.state_dict())
                    del base_model

                if 0 in ks:
                    X_train_full, y_train_full = build_windows(train_full, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
                    x_mean, x_std = fit_scaler(X_train_full)
                    y_mean, y_std = fit_scaler_1d(y_train_full)
                    X_test = apply_scaler(X_test_raw, x_mean, x_std)
                    y_test = apply_scaler_1d(y_test_raw, y_mean, y_std)
                    test_loader = make_dataloader(X_test, y_test, cfg.train.batch_size, shuffle=False)
                    for method_dir, method_name in [("M1", "M1_source_only"), ("M3", "M3_pretrain_ft")]:
                        run_dir = out_root / target_id / model_type / "k0" / f"seed{seed}" / method_dir
                        if not (run_dir / "metrics.json").exists():
                            model = make_model(model_type, len(feature_cols), cfg.data.window_size, cfg).to(device)
                            model.load_state_dict(pretrained_state)
                            eval_and_save(
                                model, test_loader, (y_mean, y_std), cfg,
                                {"method": method_name, "model": model_type, "target": target_id, "k": 0, "seed": seed},
                                run_dir,
                                history={"pretrain": hist_pt},
                                manifest={"method": method_name, "reused_pretrain": str(pretrained_state_path)},
                            )
                            del model

                for k in [v for v in ks if v > 0]:
                    train_few = train_full.iloc[:k * 24]
                    X_few_raw, y_few_raw = build_windows(train_few, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
                    if len(X_few_raw) < 10:
                        continue
                    x_mean, x_std = fit_scaler(X_few_raw)
                    y_mean, y_std = fit_scaler_1d(y_few_raw)
                    X_few = apply_scaler(X_few_raw, x_mean, x_std)
                    y_few = apply_scaler_1d(y_few_raw, y_mean, y_std)
                    X_val = apply_scaler(X_val_raw, x_mean, x_std)
                    y_val = apply_scaler_1d(y_val_raw, y_mean, y_std)
                    X_test = apply_scaler(X_test_raw, x_mean, x_std)
                    y_test = apply_scaler_1d(y_test_raw, y_mean, y_std)
                    few_bs = max(4, min(cfg.train.batch_size, len(X_few) // 4))
                    train_loader = make_dataloader(X_few, y_few, few_bs, shuffle=True)
                    val_loader = make_dataloader(X_val, y_val, few_bs, shuffle=False)
                    test_loader = make_dataloader(X_test, y_test, cfg.train.batch_size, shuffle=False)

                    for method_dir, method_name in [("M2", "M2_target_only"), ("M3", "M3_pretrain_ft")]:
                        run_dir = out_root / target_id / model_type / f"k{k}" / f"seed{seed}" / method_dir
                        if (run_dir / "metrics.json").exists():
                            continue
                        set_seed(seed)
                        model = make_model(model_type, len(feature_cols), cfg.data.window_size, cfg).to(device)
                        if method_dir == "M3":
                            model.load_state_dict(pretrained_state)
                            hist_ft = fine_tune(model, train_loader, val_loader, cfg, device, verbose=False)
                            history = {"pretrain": hist_pt, "finetune": hist_ft}
                        else:
                            history = train_model(model, train_loader, val_loader, cfg, device, verbose=False)
                        eval_and_save(
                            model, test_loader, (y_mean, y_std), cfg,
                            {"method": method_name, "model": model_type, "target": target_id, "k": k, "seed": seed},
                            run_dir,
                            history=history,
                            manifest={"method": method_name, "reused_pretrain": str(pretrained_state_path) if method_dir == "M3" else None},
                        )
                        del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    aggregate(out_root, building_ids, args.models, ks, args.seeds)


if __name__ == "__main__":
    main()
