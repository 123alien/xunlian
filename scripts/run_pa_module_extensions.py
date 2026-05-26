#!/usr/bin/env python
"""Pilot extra modules for persistence-anchored cold-start forecasting.

This script intentionally stays close to the existing PARBoost pilot but adds
top-conference-inspired modules:

- residual RevIN: per-window reversible residual scaling.
- residual gate: a conservative target few-shot gate for correction strength.
- multi-scale residual features: TimeMixer/N-HiTS-style short/medium/daily
  residual summaries.
- daily-profile similarity: PatchTST-style daily residual profile matching for
  source selection.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_par_gbdt_pilot import (  # noqa: E402
    ENERGY_IDX,
    building_arrays,
    building_stats,
    calibrate_alpha,
    make_direct_model,
    metric_row,
    save_prediction,
    signature_from_train,
    source_weights,
    split_building,
    tabularize,
    uniform_source_weights,
)
from src.config import Config  # noqa: E402
from src.phase2 import FEATURE_COLS  # noqa: E402
from src.train import build_windows  # noqa: E402


def safe_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    std = float(np.std(values))
    return max(iqr / 1.349 if iqr > 0 else 0.0, std, 1e-3)


def normalize_windows_by_building(X: np.ndarray, stats: dict) -> np.ndarray:
    Xn = X.astype(np.float32).copy()
    Xn[:, :, ENERGY_IDX] = (Xn[:, :, ENERGY_IDX] - stats["energy_mean"]) / stats["energy_scale"]
    return Xn


def window_revin_transform(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return instance-normalized windows and per-window residual scales."""
    Xn = X_raw.astype(np.float32).copy()
    e = X_raw[:, :, ENERGY_IDX].astype(float)
    mu = e.mean(axis=1, keepdims=True)
    sd = e.std(axis=1, keepdims=True)
    diffs = np.diff(e, axis=1)
    diff_scale = np.mean(np.abs(diffs), axis=1) if diffs.shape[1] else sd[:, 0]
    scale = np.maximum.reduce([sd[:, 0], diff_scale, np.full(len(e), 1e-3)])
    Xn[:, :, ENERGY_IDX] = ((e - mu) / scale[:, None]).astype(np.float32)
    return Xn, scale.astype(float)


def multiscale_residual_features(X_energy: np.ndarray, scales: tuple[int, ...] = (3, 6, 12, 24)) -> np.ndarray:
    """Compact multi-scale summaries of the energy history in each window."""
    e = np.asarray(X_energy, dtype=float)
    last = e[:, -1]
    feats = []
    for scale in scales:
        span = min(scale, e.shape[1])
        seg = e[:, -span:]
        diffs = np.diff(seg, axis=1)
        if diffs.shape[1] == 0:
            diff_mean = np.zeros(len(e), dtype=float)
            diff_abs = np.zeros(len(e), dtype=float)
        else:
            diff_mean = diffs.mean(axis=1)
            diff_abs = np.mean(np.abs(diffs), axis=1)
        feats.extend([
            last - seg.mean(axis=1),
            seg.std(axis=1),
            diff_mean,
            diff_abs,
        ])
    return np.column_stack(feats).astype(np.float32)


def append_msr_features(X_tab: np.ndarray, X_for_features: np.ndarray, use_msr: bool) -> np.ndarray:
    if not use_msr:
        return X_tab
    msr = multiscale_residual_features(X_for_features[:, :, ENERGY_IDX])
    return np.concatenate([X_tab, msr], axis=1)


def building_arrays_ext(data: pd.DataFrame, bid: str, feature_cols: list[str], cfg: Config,
                        k: int | None = None, split: str = "train", mode: str = "anchor",
                        use_msr: bool = False):
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
    last = X_raw[:, -1, ENERGY_IDX]
    if mode == "revin":
        Xn, scale = window_revin_transform(X_raw)
        X = append_msr_features(tabularize(Xn), Xn, use_msr)
        target = (y_raw - last) / scale
        return X, target, last, y_raw, stats, scale
    Xn = normalize_windows_by_building(X_raw, stats)
    X = append_msr_features(tabularize(Xn), Xn, use_msr)
    target = (y_raw - last) / stats["residual_scale"]
    scale = np.full(len(y_raw), stats["residual_scale"], dtype=float)
    return X, target, last, y_raw, stats, scale


def residual_profile_signature(frame: pd.DataFrame) -> np.ndarray | None:
    """Hourly residual profile from allowed train/few-shot data only."""
    if frame is None or len(frame) < 30 or "timestamp" not in frame.columns:
        return None
    tmp = frame[["timestamp", "energy"]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    tmp["resid"] = tmp["energy"].diff()
    tmp = tmp.dropna(subset=["resid"])
    if tmp.empty:
        return None
    tmp["hour"] = tmp["timestamp"].dt.hour
    prof = tmp.groupby("hour")["resid"].agg(["mean", "std"]).reindex(range(24)).fillna(0.0)
    vec = np.concatenate([prof["mean"].to_numpy(), prof["std"].to_numpy()])
    scale = safe_scale(vec)
    return (vec / scale).astype(float)


def source_weights_profile(data: pd.DataFrame, source_ids: list[str], target_id: str, k: int,
                           tau: float = 1.0, top_k: int | None = None) -> dict[str, float]:
    train_t, _, _ = split_building(data, target_id)
    target_sig = residual_profile_signature(train_t.iloc[: k * 24])
    if target_sig is None:
        return source_weights(data, source_ids, target_id, k, tau=tau, top_k=top_k)
    distances = []
    for sid in source_ids:
        train_s, _, _ = split_building(data, sid)
        sig = residual_profile_signature(train_s)
        if sig is None:
            continue
        distances.append((sid, float(np.linalg.norm(sig - target_sig))))
    if not distances:
        return uniform_source_weights(source_ids)
    distances.sort(key=lambda item: item[1])
    if top_k is not None and top_k > 0:
        keep = set(sid for sid, _ in distances[:top_k])
        distances = [(sid, dist) for sid, dist in distances if sid in keep]
    vals = np.array([dist for _, dist in distances], dtype=float)
    denom = np.median(vals) + 1e-6
    raw = np.exp(-vals / max(tau * denom, 1e-6))
    raw = raw / max(raw.sum(), 1e-12)
    return {sid: float(w) for (sid, _), w in zip(distances, raw)}


def source_matrix_ext(data: pd.DataFrame, source_ids: list[str], target_id: str, k: int,
                      feature_cols: list[str], cfg: Config, max_source_windows: int,
                      seed: int, tau: float, top_source_k: int | None,
                      use_similarity: bool, mode: str, use_msr: bool = False,
                      similarity_kind: str = "stats",
                      source_cache: dict[tuple[str, str, bool], tuple] | None = None):
    if not use_similarity:
        weights = uniform_source_weights(source_ids)
    elif similarity_kind == "profile":
        weights = source_weights_profile(data, source_ids, target_id, k, tau=tau, top_k=top_source_k)
    else:
        weights = source_weights(data, source_ids, target_id, k, tau=tau, top_k=top_source_k)
    Xs, ys, ws, used = [], [], [], []
    rng = np.random.default_rng(seed)
    for sid in source_ids:
        cache_key = (sid, mode, bool(use_msr))
        if source_cache is not None and cache_key in source_cache:
            arr = source_cache[cache_key]
        else:
            arr = building_arrays_ext(data, sid, feature_cols, cfg, k=None, split="train", mode=mode, use_msr=use_msr)
            if source_cache is not None and arr is not None:
                source_cache[cache_key] = arr
        if arr is None:
            continue
        X, target, _, _, stats, _ = arr
        if len(X) > max_source_windows:
            idx = rng.choice(len(X), size=max_source_windows, replace=False)
            X, target = X[idx], target[idx]
        base_w = weights.get(sid, 0.0)
        reliability = 1.0 / (1.0 + stats["zero_ratio"])
        Xs.append(X)
        ys.append(target)
        ws.append(np.full(len(X), base_w * reliability, dtype=float))
        used.append(sid)
    if not Xs:
        return None
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    w_all = np.concatenate(ws, axis=0)
    w_all = w_all / max(np.mean(w_all), 1e-12)
    return X_all, y_all, w_all, weights, used


def fewshot_gate(data: pd.DataFrame, target_id: str, k: int) -> float:
    sig = signature_from_train(data, target_id, k_days=k)
    if sig is None:
        return 0.75
    zero_ratio = float(sig[2])
    activity = float(sig[4])
    # Conservative correction for inactive or extremely smooth few-shot targets.
    gate = 0.25 + 0.85 * activity
    gate *= 1.0 - 0.65 * zero_ratio
    return float(np.clip(gate, 0.05, 1.0))


def aggregate(rows: list[dict], out: Path):
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "table_pa_module_extensions.csv", index=False)
    if df.empty:
        return
    summary = df.groupby(["method", "k"]).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        worst_MAE=("MAE", "max"),
        mean_RMSE=("RMSE", "mean"),
        n=("MAE", "size"),
    ).reset_index().sort_values(["k", "mean_MAE"])
    summary.to_csv(out / "table_pa_module_extensions_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly_expanded.csv")
    ap.add_argument("--output-dir", default="results/pa_module_extensions_bdg2")
    ap.add_argument("--building-manifest", default=None)
    ap.add_argument("--n-buildings", type=int, default=24)
    ap.add_argument("--k", type=int, nargs="+", default=[3, 7, 14, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-source-windows", type=int, default=800)
    ap.add_argument("--target-weight", type=float, default=25.0)
    ap.add_argument("--tree-estimators", type=int, default=160)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--top-source-k", type=int, default=5)
    ap.add_argument("--alpha-grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
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
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    if args.shard_count > 1:
        building_ids = [bid for i, bid in enumerate(building_ids) if i % args.shard_count == args.shard_index]
    out = Path(args.output_dir)
    rows = []
    source_cache = {}

    variants = [
        {"method": "M1_ANCHOR", "mode": "anchor", "similarity": False, "similarity_kind": "stats", "calibrate": False, "gate": False, "msr": False},
        {"method": "M2_ANCHOR_SIM", "mode": "anchor", "similarity": True, "similarity_kind": "stats", "calibrate": False, "gate": False, "msr": False},
        {"method": "M3_ANCHOR_REVIN", "mode": "revin", "similarity": False, "similarity_kind": "stats", "calibrate": False, "gate": False, "msr": False},
        {"method": "M4_ANCHOR_GATE", "mode": "anchor", "similarity": False, "similarity_kind": "stats", "calibrate": False, "gate": True, "msr": False},
        {"method": "M5_ANCHOR_REVIN_SIM", "mode": "revin", "similarity": True, "similarity_kind": "stats", "calibrate": False, "gate": False, "msr": False},
        {"method": "M6_FULL_REVIN_SIM_GATE", "mode": "revin", "similarity": True, "similarity_kind": "stats", "calibrate": False, "gate": True, "msr": False},
        {"method": "M7_FULL_REVIN_SIM_CAL", "mode": "revin", "similarity": True, "similarity_kind": "stats", "calibrate": True, "gate": False, "msr": False},
        {"method": "M8_ANCHOR_MSR", "mode": "anchor", "similarity": False, "similarity_kind": "stats", "calibrate": False, "gate": False, "msr": True},
        {"method": "M9_REVIN_MSR", "mode": "revin", "similarity": False, "similarity_kind": "stats", "calibrate": False, "gate": False, "msr": True},
        {"method": "M10_REVIN_MSR_DPS", "mode": "revin", "similarity": True, "similarity_kind": "profile", "calibrate": False, "gate": False, "msr": True},
        {"method": "M11_REVIN_MSR_DPS_CAL", "mode": "revin", "similarity": True, "similarity_kind": "profile", "calibrate": True, "gate": False, "msr": True},
    ]
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in variants if v["method"] in wanted]
        if not variants:
            raise ValueError(f"No variants selected from {sorted(wanted)}")

    for target_id in building_ids:
        source_ids = [b for b in building_ids if b != target_id]
        for k in args.k:
            for seed in args.seeds:
                for spec in variants:
                    method = spec["method"]
                    run_dir = out / target_id / f"k{k}" / f"seed{seed}" / method
                    metrics_file = run_dir / "metrics.json"
                    if metrics_file.exists():
                        rows.append(json.loads(metrics_file.read_text()))
                        continue
                    train_t = building_arrays_ext(data, target_id, feature_cols, cfg, k=k, split="train", mode=spec["mode"], use_msr=spec["msr"])
                    val_t = building_arrays_ext(data, target_id, feature_cols, cfg, k=k, split="val", mode=spec["mode"], use_msr=spec["msr"])
                    test_t = building_arrays_ext(data, target_id, feature_cols, cfg, k=k, split="test", mode=spec["mode"], use_msr=spec["msr"])
                    if train_t is None or test_t is None or len(train_t[0]) < 5:
                        continue
                    src = source_matrix_ext(
                        data, source_ids, target_id, k, feature_cols, cfg,
                        args.max_source_windows, seed, args.tau, args.top_source_k,
                        use_similarity=spec["similarity"], mode=spec["mode"], use_msr=spec["msr"],
                        similarity_kind=spec["similarity_kind"],
                        source_cache=source_cache,
                    )
                    if src is None:
                        continue
                    X_s, y_s, w_s, sim_weights, used_sources = src
                    X_t, y_t, _, _, _, _ = train_t
                    X_test, _, last_test, y_test, target_stats, scale_test = test_t
                    X_train = np.concatenate([X_s, X_t], axis=0)
                    y_train = np.concatenate([y_s, y_t], axis=0)
                    w_train = np.concatenate([w_s, np.full(len(X_t), args.target_weight, dtype=float)])
                    model = make_direct_model(seed, args.tree_estimators)
                    model.fit(X_train, y_train, sample_weight=w_train)
                    alpha = 1.0
                    alpha_val_mae = None
                    if spec["calibrate"] and val_t is not None:
                        X_val, _, last_val, y_val, _, scale_val = val_t
                        # calibrate_alpha assumes scalar scale, so do a local vector-scale grid.
                        pred_val = model.predict(X_val)
                        best = (1.0, float("inf"))
                        for a in args.alpha_grid:
                            mae = float(np.mean(np.abs(y_val - (last_val + a * pred_val * scale_val))))
                            if mae < best[1]:
                                best = (float(a), mae)
                        alpha, alpha_val_mae = best
                    if spec["gate"]:
                        alpha *= fewshot_gate(data, target_id, k)
                    pred_resid = model.predict(X_test)
                    y_pred = last_test + alpha * pred_resid * scale_test
                    row = metric_row(y_test, y_pred, cfg, {
                        "method": method,
                        "target": target_id,
                        "k": int(k),
                        "seed": int(seed),
                        "mode": spec["mode"],
                        "similarity": bool(spec["similarity"]),
                        "similarity_kind": spec["similarity_kind"],
                        "msr": bool(spec["msr"]),
                        "gate": bool(spec["gate"]),
                        "calibrate": bool(spec["calibrate"]),
                        "alpha": float(alpha),
                        "alpha_val_mae": alpha_val_mae,
                    })
                    run_dir.mkdir(parents=True, exist_ok=True)
                    metrics_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    save_prediction(run_dir, y_test, y_pred, {
                        "variant": spec,
                        "similarity_weights": sim_weights,
                        "used_sources": used_sources,
                        "alpha": alpha,
                        "alpha_val_mae": alpha_val_mae,
                    })
                    rows.append(row)
                    print(f"[PA-MOD] {target_id} k{k} seed{seed} {method}", flush=True)

    aggregate(rows, out / "aggregate")


if __name__ == "__main__":
    main()
