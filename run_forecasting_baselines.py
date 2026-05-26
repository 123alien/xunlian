#!/usr/bin/env python
"""Run classical forecasting baselines for the forecasting-focused paper."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.train import build_windows  # noqa: E402
from src.anomaly import rolling_mad_scores  # noqa: E402


FEATURE_COLS = ["energy", "hour", "day_of_week", "is_weekend", "month"]
K_VALUES = [3, 7, 14]
SEEDS = [42, 43, 44, 45, 46]


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def sample_rows(X: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx], y[idx]


def make_regressor(name: str, seed: int, tree_estimators: int = 80):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=tree_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=tree_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gbdt":
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown regressor: {name}")


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, meta: dict, cfg: Config) -> dict:
    err = np.abs(y_true - y_pred)
    scores = rolling_mad_scores(err, cfg.anomaly.mad_window, cfg.anomaly.epsilon)
    row = compute_all(y_true, y_pred, scores)
    row.update(meta)
    return row


def prepare_target(data: pd.DataFrame, target_id: str, feature_cols: list[str], cfg: Config):
    tgt = data[data["building_id"] == target_id].sort_values("timestamp")
    n = len(tgt)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_full = tgt.iloc[:train_end]
    test = tgt.iloc[val_end:]
    X_test, y_test = build_windows(test, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
    if len(X_test) == 0:
        return None
    return train_full, X_test, y_test


def source_windows(data: pd.DataFrame, source_ids: list[str], feature_cols: list[str], cfg: Config):
    Xs, ys = [], []
    for bid in source_ids:
        b = data[data["building_id"] == bid].sort_values("timestamp")
        train_end = int(len(b) * 0.6)
        train = b.iloc[:train_end]
        X, y = build_windows(train, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon)
        if len(X):
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return None, None
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def run_naive_baselines(X_test: np.ndarray, y_test: np.ndarray, cfg: Config, meta_base: dict) -> list[dict]:
    energy_idx = 0
    rows = []
    preds = {
        "persistence": X_test[:, -1, energy_idx],
        "seasonal_naive_24": X_test[:, 0, energy_idx],
    }
    for name, pred in preds.items():
        rows.append(metric_row(y_test, pred, {**meta_base, "method": name}, cfg))
    return rows


def run_ml_baseline(name: str, mode: str, seed: int,
                    X_target: np.ndarray, y_target: np.ndarray,
                    X_source: np.ndarray | None, y_source: np.ndarray | None,
                    X_test: np.ndarray, y_test: np.ndarray,
                    max_source_windows: int, tree_estimators: int,
                    cfg: Config, meta_base: dict) -> dict | None:
    if mode == "target_only":
        X_train, y_train = X_target, y_target
    elif mode == "source_target":
        if X_source is None or y_source is None:
            return None
        Xs, ys = sample_rows(X_source, y_source, max_source_windows, seed)
        X_train = np.concatenate([Xs, X_target], axis=0)
        y_train = np.concatenate([ys, y_target], axis=0)
    else:
        raise ValueError(mode)

    if len(X_train) < 5:
        return None
    model = make_regressor(name, seed, tree_estimators=tree_estimators)
    model.fit(flatten(X_train), y_train)
    pred = model.predict(flatten(X_test))
    return metric_row(
        y_test,
        pred,
        {**meta_base, "method": f"{name}_{mode}", "seed": seed},
        cfg,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    ap.add_argument("--output-dir", default="results/phase2_paper/forecasting_baselines")
    ap.add_argument("--n-buildings", type=int, default=12)
    ap.add_argument("--building-manifest", default=None)
    ap.add_argument("--k", type=int, nargs="+", default=K_VALUES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--models", nargs="+", default=["ridge", "random_forest", "extra_trees", "hist_gbdt"])
    ap.add_argument("--max-source-windows", type=int, default=5000)
    ap.add_argument("--tree-estimators", type=int, default=80)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    if args.building_manifest:
        manifest = pd.read_csv(args.building_manifest)
        building_ids = manifest["building_id"].dropna().astype(str).tolist()[:args.n_buildings]
    else:
        building_ids = sorted(data["building_id"].unique())[:args.n_buildings]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for target_id in building_ids:
        prepared = prepare_target(data, target_id, feature_cols, cfg)
        if prepared is None:
            continue
        train_full, X_test, y_test = prepared
        source_ids = [b for b in building_ids if b != target_id]
        X_source, y_source = source_windows(data, source_ids, feature_cols, cfg)

        for k in args.k:
            train_few = train_full.iloc[:k * 24]
            X_target, y_target = build_windows(
                train_few, feature_cols, "energy", cfg.data.window_size, cfg.data.horizon
            )
            if len(X_target) < 5:
                continue
            meta_base = {"target": target_id, "k": int(k)}
            for seed in args.seeds:
                rows.extend(run_naive_baselines(
                    X_test, y_test, cfg, {**meta_base, "seed": seed}
                ))
                for model_name in args.models:
                    for mode in ["target_only", "source_target"]:
                        row = run_ml_baseline(
                            model_name, mode, seed,
                            X_target, y_target,
                            X_source, y_source,
                            X_test, y_test,
                            args.max_source_windows,
                            args.tree_estimators,
                            cfg,
                            meta_base,
                        )
                        if row is not None:
                            rows.append(row)
                print(f"[forecasting-baselines] {target_id} k{k} seed{seed}", flush=True)

    df = pd.DataFrame(rows)
    detail = out / "table_forecasting_baselines.csv"
    df.to_csv(detail, index=False)
    if not df.empty:
        summary = df.groupby(["method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
        summary.to_csv(out / "table_forecasting_baselines_summary.csv", index=False)
        by_target = df.groupby(["target", "method", "k"])[["MAE", "RMSE", "sMAPE", "sigma_err"]].mean().reset_index()
        by_target.to_csv(out / "table_forecasting_baselines_building_level.csv", index=False)
        print(summary.sort_values(["k", "MAE"]).to_string(index=False))
    print(f"[forecasting-baselines] saved {detail} rows={len(df)}")


if __name__ == "__main__":
    main()
