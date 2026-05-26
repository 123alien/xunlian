#!/usr/bin/env python
"""Persistence-anchored residual tree transfer pilot.

The model predicts a scale-normalized residual instead of raw next-step load:

    y_hat[t+1] = y[t] + residual_scale(target) * f(features)

Source samples are weighted by source-target similarity computed only from
target few-shot data. Target few-shot samples receive an explicit weight boost.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anomaly import rolling_mad_scores  # noqa: E402
from src.config import Config  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.phase2 import FEATURE_COLS  # noqa: E402
from src.train import build_windows  # noqa: E402


ENERGY_IDX = 0


def split_building(data: pd.DataFrame, bid: str):
    g = data[data["building_id"] == bid].sort_values("timestamp")
    n = len(g)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return g.iloc[:train_end], g.iloc[train_end:val_end], g.iloc[val_end:]


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    std = float(np.std(values))
    scale = max(iqr / 1.349 if iqr > 0 else 0.0, std, 1e-3)
    return scale


def building_stats(train: pd.DataFrame) -> dict:
    e = train["energy"].to_numpy(dtype=float)
    diff = np.diff(e) if len(e) > 1 else np.array([0.0])
    return {
        "energy_mean": float(np.mean(e)) if len(e) else 0.0,
        "energy_scale": robust_scale(e),
        "residual_scale": robust_scale(diff),
        "zero_ratio": float(np.mean(np.isclose(e, 0.0))) if len(e) else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
    }


def normalize_windows(X: np.ndarray, stats: dict) -> np.ndarray:
    Xn = X.astype(np.float32).copy()
    Xn[:, :, ENERGY_IDX] = (Xn[:, :, ENERGY_IDX] - stats["energy_mean"]) / stats["energy_scale"]
    return Xn


def tabularize(X: np.ndarray) -> np.ndarray:
    energy = X[:, :, ENERGY_IDX]
    diffs = np.diff(energy, axis=1)
    stats = np.column_stack([
        energy[:, -1],
        energy.mean(axis=1),
        energy.std(axis=1),
        energy.min(axis=1),
        energy.max(axis=1),
        diffs[:, -1] if diffs.shape[1] else np.zeros(len(X)),
        np.mean(np.abs(diffs), axis=1) if diffs.shape[1] else np.zeros(len(X)),
    ])
    return np.concatenate([X.reshape(X.shape[0], -1), diffs, stats], axis=1)


def residual_target(X_raw: np.ndarray, y_raw: np.ndarray, stats: dict) -> np.ndarray:
    return (y_raw - X_raw[:, -1, ENERGY_IDX]) / stats["residual_scale"]


def signature_from_train(data: pd.DataFrame, bid: str, k_days: int | None = None):
    train, _, _ = split_building(data, bid)
    if k_days is not None:
        train = train.iloc[: k_days * 24]
    e = train["energy"].to_numpy(dtype=float)
    if len(e) < 2:
        return None
    by_hour = train.groupby(train["timestamp"].dt.hour)["energy"].mean().reindex(range(24)).fillna(0).to_numpy()
    diff = np.diff(e)
    scale = max(np.mean(e), robust_scale(e), 1e-6)
    return np.concatenate([
        np.array([
            np.mean(e) / scale,
            np.std(e) / scale,
            np.mean(np.isclose(e, 0.0)),
            (np.percentile(e, 95) - np.percentile(e, 5)) / scale,
            np.mean(np.abs(diff)) / scale if len(diff) else 0.0,
        ]),
        by_hour / scale,
    ])


def source_weights(data: pd.DataFrame, source_ids: list[str], target_id: str, k: int,
                   tau: float = 1.0, top_k: int | None = None) -> dict[str, float]:
    target_sig = signature_from_train(data, target_id, k_days=k)
    rows, sigs = [], []
    for sid in source_ids:
        sig = signature_from_train(data, sid)
        if sig is not None:
            rows.append(sid)
            sigs.append(sig)
    if target_sig is None or not sigs:
        return {sid: 1.0 / max(len(source_ids), 1) for sid in source_ids}
    mat = np.vstack(sigs)
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd < 1e-8] = 1.0
    d = np.linalg.norm((mat - mu) / sd - (target_sig - mu) / sd, axis=1)
    if top_k is not None and top_k < len(d):
        keep = np.argsort(d)[:top_k]
        mask = np.zeros_like(d, dtype=bool)
        mask[keep] = True
        d = np.where(mask, d, np.inf)
    finite = np.isfinite(d)
    score = np.full_like(d, -np.inf, dtype=float)
    score[finite] = -d[finite] / max(tau, 1e-6)
    score[finite] -= np.max(score[finite])
    w = np.exp(score)
    w[~np.isfinite(w)] = 0.0
    w = w / max(w.sum(), 1e-12)
    return {sid: float(wi) for sid, wi in zip(rows, w)}


def uniform_source_weights(source_ids: list[str]) -> dict[str, float]:
    if not source_ids:
        return {}
    return {sid: 1.0 / len(source_ids) for sid in source_ids}


def building_arrays(data: pd.DataFrame, bid: str, feature_cols: list[str], cfg: Config,
                    k: int | None = None, split: str = "train"):
    train, val, test = split_building(data, bid)
    if split == "train":
        frame = train if k is None else train.iloc[: k * 24]
        stats_frame = frame
    elif split == "val":
        frame = val
        stats_frame = train if k is None else train.iloc[: k * 24]
    elif split == "test":
        frame = test
        stats_frame = train if k is None else train.iloc[: k * 24]
    else:
        raise ValueError(split)
    X_raw, y_raw = build_windows(frame, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    if len(X_raw) == 0:
        return None
    stats = building_stats(stats_frame)
    X = tabularize(normalize_windows(X_raw, stats))
    r = residual_target(X_raw, y_raw, stats)
    return X, r, X_raw[:, -1, ENERGY_IDX], y_raw, stats


def source_matrix(data: pd.DataFrame, source_ids: list[str], target_id: str, k: int,
                  feature_cols: list[str], cfg: Config, max_source_windows: int,
                  seed: int, tau: float, top_source_k: int | None,
                  use_similarity: bool = True):
    if use_similarity:
        weights = source_weights(data, source_ids, target_id, k, tau=tau, top_k=top_source_k)
    else:
        weights = uniform_source_weights(source_ids)
    Xs, rs, ws, ids = [], [], [], []
    rng = np.random.default_rng(seed)
    for sid in source_ids:
        arr = building_arrays(data, sid, feature_cols, cfg, k=None, split="train")
        if arr is None:
            continue
        X, r, _, _, stats = arr
        if len(X) > max_source_windows:
            idx = rng.choice(len(X), size=max_source_windows, replace=False)
            X, r = X[idx], r[idx]
        base_w = weights.get(sid, 0.0)
        reliability = 1.0 / (1.0 + stats["zero_ratio"])
        Xs.append(X)
        rs.append(r)
        ws.append(np.full(len(X), base_w * reliability, dtype=float))
        ids.append(sid)
    if not Xs:
        return None
    X_all = np.concatenate(Xs, axis=0)
    r_all = np.concatenate(rs, axis=0)
    w_all = np.concatenate(ws, axis=0)
    w_all = w_all / max(np.mean(w_all), 1e-12)
    return X_all, r_all, w_all, weights, ids


def make_model(name: str, seed: int, estimators: int):
    name = name.replace("_cal", "")
    if name == "par_extra_trees_sw":
        return ExtraTreesRegressor(
            n_estimators=estimators,
            min_samples_leaf=2,
            max_features=0.75,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "par_random_forest_sw":
        return RandomForestRegressor(
            n_estimators=estimators,
            min_samples_leaf=2,
            max_features=0.75,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "par_hist_gbdt_sw":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=500,
            learning_rate=0.035,
            l2_regularization=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            random_state=seed,
        )
    raise ValueError(name)


def make_direct_model(seed: int, estimators: int):
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=estimators,
        learning_rate=0.035,
        l2_regularization=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        random_state=seed,
    )


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config, meta: dict) -> dict:
    err = np.abs(y_true - y_pred)
    scores = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
    row = compute_all(y_true, y_pred, scores)
    row.update(meta)
    return row


def save_prediction(run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, manifest: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def calibrate_alpha(model, X_val: np.ndarray, last_val: np.ndarray, y_val: np.ndarray,
                    residual_scale: float, grid: list[float]) -> tuple[float, float]:
    r_pred = model.predict(X_val)
    best_alpha = 1.0
    best_mae = float("inf")
    for alpha in grid:
        pred = last_val + float(alpha) * r_pred * residual_scale
        mae = float(np.mean(np.abs(y_val - pred)))
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    return best_alpha, best_mae


def direct_source_matrix(data: pd.DataFrame, source_ids: list[str], target_id: str, k: int,
                         feature_cols: list[str], cfg: Config, max_source_windows: int,
                         seed: int, tau: float, top_source_k: int | None,
                         use_similarity: bool = True):
    weights = source_weights(data, source_ids, target_id, k, tau=tau, top_k=top_source_k) if use_similarity else uniform_source_weights(source_ids)
    Xs, ys, ws, ids = [], [], [], []
    rng = np.random.default_rng(seed)
    for sid in source_ids:
        arr = building_arrays(data, sid, feature_cols, cfg, k=None, split="train")
        if arr is None:
            continue
        X, _, _, y, stats = arr
        if len(X) > max_source_windows:
            idx = rng.choice(len(X), size=max_source_windows, replace=False)
            X, y = X[idx], y[idx]
        base_w = weights.get(sid, 0.0)
        reliability = 1.0 / (1.0 + stats["zero_ratio"])
        Xs.append(X)
        ys.append(y)
        ws.append(np.full(len(X), base_w * reliability, dtype=float))
        ids.append(sid)
    if not Xs:
        return None
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    w_all = np.concatenate(ws, axis=0)
    w_all = w_all / max(np.mean(w_all), 1e-12)
    return X_all, y_all, w_all, weights, ids


def aggregate(rows: list[dict], out: Path):
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "table_par_gbdt_pilot.csv", index=False)
    if df.empty:
        return
    by_target = df.groupby(["target", "method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
    by_target.to_csv(out / "table_par_gbdt_pilot_building_level.csv", index=False)
    summary = df.groupby(["method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
    summary.to_csv(out / "table_par_gbdt_pilot_summary.csv", index=False)
    print(summary.sort_values(["k", "MAE"]).to_string(index=False))
    print(f"[PAR-GBDT] rows={len(df)} saved={out / 'table_par_gbdt_pilot.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly_expanded.csv")
    ap.add_argument("--output-dir", default="results/par_gbdt_pilot_bdg2")
    ap.add_argument("--building-manifest", default=None)
    ap.add_argument("--n-buildings", type=int, default=16)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 14, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=["par_extra_trees_sw", "par_hist_gbdt_sw"])
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--alpha-grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25])
    ap.add_argument("--max-source-windows", type=int, default=1200)
    ap.add_argument("--target-weight", type=float, default=25.0)
    ap.add_argument("--tree-estimators", type=int, default=240)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--top-source-k", type=int, default=5)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    if args.building_manifest:
        manifest = pd.read_csv(args.building_manifest)
        id_col = "building_id" if "building_id" in manifest.columns else "target"
        building_ids = manifest[id_col].dropna().astype(str).tolist()[: args.n_buildings]
    else:
        building_ids = sorted(data["building_id"].dropna().astype(str).unique().tolist())[: args.n_buildings]
    out = Path(args.output_dir)

    rows = []
    for target_id in building_ids:
        source_ids = [b for b in building_ids if b != target_id]
        for k in args.k:
            target_train = building_arrays(data, target_id, feature_cols, cfg, k=k, split="train")
            target_val = building_arrays(data, target_id, feature_cols, cfg, k=k, split="val")
            target_test = building_arrays(data, target_id, feature_cols, cfg, k=k, split="test")
            if target_train is None or target_test is None or len(target_train[0]) < 5:
                continue
            X_t, r_t, _, _, target_stats = target_train
            X_val = last_val = y_val = None
            if target_val is not None:
                X_val, _, last_val, y_val, _ = target_val
            X_test, _, last_test, y_test, _ = target_test

            for seed in args.seeds:
                src = source_matrix(
                    data, source_ids, target_id, k, feature_cols, cfg,
                    args.max_source_windows, seed, args.tau, args.top_source_k,
                )
                if src is None:
                    continue
                X_s, r_s, w_s, sim_weights, used_sources = src
                X_train = np.concatenate([X_s, X_t], axis=0)
                r_train = np.concatenate([r_s, r_t], axis=0)
                w_train = np.concatenate([w_s, np.full(len(X_t), args.target_weight, dtype=float)], axis=0)

                for model_name in args.models:
                    calibrate = model_name.endswith("_cal")
                    base_model_name = model_name.replace("_cal", "")
                    method = model_name.upper()
                    run_dir = out / target_id / f"k{k}" / f"seed{seed}" / method
                    metrics_file = run_dir / "metrics.json"
                    if metrics_file.exists():
                        rows.append(json.loads(metrics_file.read_text()))
                        continue
                    model = make_model(base_model_name, seed, args.tree_estimators)
                    model.fit(X_train, r_train, sample_weight=w_train)
                    alpha = 1.0
                    alpha_val_mae = None
                    if calibrate and X_val is not None and len(X_val):
                        alpha, alpha_val_mae = calibrate_alpha(
                            model, X_val, last_val, y_val, target_stats["residual_scale"], args.alpha_grid
                        )
                    r_pred = model.predict(X_test)
                    y_pred = last_test + alpha * r_pred * target_stats["residual_scale"]
                    row = metric_row(y_test, y_pred, cfg, {
                        "method": method,
                        "model": model_name,
                        "target": target_id,
                        "k": int(k),
                        "seed": int(seed),
                        "target_weight": float(args.target_weight),
                        "max_source_windows": int(args.max_source_windows),
                        "alpha": float(alpha),
                        "alpha_val_mae": alpha_val_mae,
                    })
                    run_dir.mkdir(parents=True, exist_ok=True)
                    metrics_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    save_prediction(run_dir, y_test, y_pred, {
                        "method": method,
                        "residual_anchor": "last_observed_energy",
                        "residual_scale": target_stats["residual_scale"],
                        "alpha": alpha,
                        "alpha_grid": args.alpha_grid,
                        "alpha_val_mae": alpha_val_mae,
                        "similarity_weights": sim_weights,
                        "used_sources": used_sources,
                        "tau": args.tau,
                        "top_source_k": args.top_source_k,
                        "target_weight": args.target_weight,
                    })
                    rows.append(row)
                    print(f"[PAR-GBDT] {target_id} k{k} seed{seed} {method}", flush=True)

                if args.ablations:
                    ablation_specs = [
                        ("ABL_DIRECT_HISTGBDT_ST", "direct", False, False),
                        ("ABL_RESIDUAL_UNIFORM", "residual", False, False),
                        ("ABL_RESIDUAL_SIM", "residual", True, False),
                        ("ABL_RESIDUAL_CAL", "residual", False, True),
                        ("ABL_FULL_PARBOOST", "residual", True, True),
                    ]
                    cached = {}
                    for method, target_mode, use_similarity, use_calibration in ablation_specs:
                        run_dir = out / target_id / f"k{k}" / f"seed{seed}" / method
                        metrics_file = run_dir / "metrics.json"
                        if metrics_file.exists():
                            rows.append(json.loads(metrics_file.read_text()))
                            continue
                        key = (target_mode, use_similarity)
                        if key not in cached:
                            if target_mode == "direct":
                                src_direct = direct_source_matrix(
                                    data, source_ids, target_id, k, feature_cols, cfg,
                                    args.max_source_windows, seed, args.tau, args.top_source_k,
                                    use_similarity=use_similarity,
                                )
                                if src_direct is None:
                                    continue
                                X_s_d, y_s_d, w_s_d, sim_weights_d, used_sources_d = src_direct
                                X_train_d = np.concatenate([X_s_d, X_t], axis=0)
                                y_train_d = np.concatenate([y_s_d, target_train[3]], axis=0)
                                w_train_d = np.concatenate([w_s_d, np.full(len(X_t), args.target_weight, dtype=float)], axis=0)
                                cached[key] = (X_train_d, y_train_d, w_train_d, sim_weights_d, used_sources_d)
                            else:
                                src_res = source_matrix(
                                    data, source_ids, target_id, k, feature_cols, cfg,
                                    args.max_source_windows, seed, args.tau, args.top_source_k,
                                    use_similarity=use_similarity,
                                )
                                if src_res is None:
                                    continue
                                X_s_r, r_s_r, w_s_r, sim_weights_r, used_sources_r = src_res
                                X_train_r = np.concatenate([X_s_r, X_t], axis=0)
                                r_train_r = np.concatenate([r_s_r, r_t], axis=0)
                                w_train_r = np.concatenate([w_s_r, np.full(len(X_t), args.target_weight, dtype=float)], axis=0)
                                cached[key] = (X_train_r, r_train_r, w_train_r, sim_weights_r, used_sources_r)
                        X_ab, y_ab, w_ab, sim_weights_ab, used_sources_ab = cached[key]
                        model = make_direct_model(seed, args.tree_estimators)
                        model.fit(X_ab, y_ab, sample_weight=w_ab)
                        alpha = 1.0
                        alpha_val_mae = None
                        if target_mode == "direct":
                            y_pred = model.predict(X_test)
                        else:
                            if use_calibration and X_val is not None and len(X_val):
                                alpha, alpha_val_mae = calibrate_alpha(
                                    model, X_val, last_val, y_val, target_stats["residual_scale"], args.alpha_grid
                                )
                            r_pred = model.predict(X_test)
                            y_pred = last_test + alpha * r_pred * target_stats["residual_scale"]
                        row = metric_row(y_test, y_pred, cfg, {
                            "method": method,
                            "model": "parboost_ablation",
                            "target": target_id,
                            "k": int(k),
                            "seed": int(seed),
                            "target_weight": float(args.target_weight),
                            "max_source_windows": int(args.max_source_windows),
                            "alpha": float(alpha),
                            "alpha_val_mae": alpha_val_mae,
                        })
                        run_dir.mkdir(parents=True, exist_ok=True)
                        metrics_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
                        save_prediction(run_dir, y_test, y_pred, {
                            "method": method,
                            "target_mode": target_mode,
                            "use_similarity": use_similarity,
                            "use_calibration": use_calibration,
                            "residual_anchor": "last_observed_energy" if target_mode == "residual" else None,
                            "residual_scale": target_stats["residual_scale"],
                            "alpha": alpha,
                            "alpha_grid": args.alpha_grid,
                            "alpha_val_mae": alpha_val_mae,
                            "similarity_weights": sim_weights_ab,
                            "used_sources": used_sources_ab,
                        })
                        rows.append(row)
                        print(f"[PAR-ABL] {target_id} k{k} seed{seed} {method}", flush=True)

    aggregate(rows, out / "aggregate")


if __name__ == "__main__":
    main()
