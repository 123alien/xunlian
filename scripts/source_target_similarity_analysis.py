#!/usr/bin/env python
"""Analyze whether source-target similarity explains transfer gains."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = [
    "mean_energy", "std_energy", "cv_energy",
    "daily_profile_range", "weekly_profile_range", "sqm",
]


def building_features(data: pd.DataFrame, building_ids: list[str]) -> pd.DataFrame:
    rows = []
    for bid in building_ids:
        g = data[data["building_id"] == bid].sort_values("timestamp")
        if g.empty:
            continue
        energy = g["energy"].astype(float)
        daily = energy.groupby(g["timestamp"].dt.hour).mean()
        weekly = energy.groupby(g["timestamp"].dt.dayofweek).mean()
        rows.append({
            "building_id": bid,
            "building_type": str(g["building_type"].dropna().iloc[0]) if "building_type" in g else "Unknown",
            "sqm": float(g["sqm"].dropna().iloc[0]) if "sqm" in g and g["sqm"].notna().any() else np.nan,
            "mean_energy": energy.mean(),
            "std_energy": energy.std(),
            "cv_energy": energy.std() / max(abs(energy.mean()), 1e-8),
            "daily_profile_range": daily.max() - daily.min(),
            "weekly_profile_range": weekly.max() - weekly.min(),
        })
    return pd.DataFrame(rows)


def similarity_table(feat: pd.DataFrame) -> pd.DataFrame:
    x = feat[FEATURES].copy()
    x = x.fillna(x.median(numeric_only=True))
    x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    mat = x.to_numpy(dtype=float)
    ids = feat["building_id"].tolist()
    rows = []
    for i, bid in enumerate(ids):
        dists = []
        for j, sid in enumerate(ids):
            if i == j:
                continue
            dists.append((sid, float(np.linalg.norm(mat[i] - mat[j]))))
        dists.sort(key=lambda z: z[1])
        rows.append({
            "target": bid,
            "nearest_source": dists[0][0],
            "nearest_distance": dists[0][1],
            "mean_source_distance": float(np.mean([d for _, d in dists])),
        })
    return pd.DataFrame(rows)


def transfer_gains(method_table: pd.DataFrame, model: str = "dlinear") -> pd.DataFrame:
    df = method_table.copy()
    if "target" not in df.columns and "building" in df.columns:
        df = df.rename(columns={"building": "target"})
    if "model" in df.columns:
        df = df[df["model"] == model]
        df = df[df["method"].isin(["M2_target_only", "M3_pretrain_ft", "M3R_replay_ft"])]
        df["canonical_method"] = df["method"]
    else:
        prefix = f"{model}_"
        df = df[df["method"].astype(str).str.startswith(prefix)].copy()
        df["canonical_method"] = df["method"].astype(str).str.replace(prefix, "", regex=False)
        df.loc[df["canonical_method"].str.startswith("M3R_lambda"), "canonical_method"] = "M3R_replay_ft"
    bld = df.groupby(["target", "k", "method"])["MAE"].mean().reset_index()
    if "canonical_method" in df.columns:
        bld = df.groupby(["target", "k", "canonical_method"])["MAE"].mean().reset_index()
        wide = bld.pivot_table(index=["target", "k"], columns="canonical_method", values="MAE", aggfunc="min").reset_index()
    else:
        wide = bld.pivot_table(index=["target", "k"], columns="method", values="MAE").reset_index()
    if {"M2_target_only", "M3_pretrain_ft"}.issubset(wide.columns):
        wide["gain_m3_vs_m2"] = wide["M2_target_only"] - wide["M3_pretrain_ft"]
    if {"M3_pretrain_ft", "M3R_replay_ft"}.issubset(wide.columns):
        wide["gain_srft_vs_m3"] = wide["M3_pretrain_ft"] - wide["M3R_replay_ft"]
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    ap.add_argument("--building-manifest", default="results/phase2_paper/expanded/building_manifest_24.csv")
    ap.add_argument("--method-table", default="results/phase2_paper/expanded/forecasting_paper/table_all_forecasting_methods.csv")
    ap.add_argument("--output-dir", default="results/phase2_paper/expanded/similarity")
    ap.add_argument("--model", default="dlinear")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, parse_dates=["timestamp"])
    manifest = pd.read_csv(args.building_manifest)
    ids = manifest["building_id"].astype(str).tolist()

    feat = building_features(data, ids)
    sim = similarity_table(feat)
    methods = pd.read_csv(args.method_table)
    gains = transfer_gains(methods, args.model)
    merged = gains.merge(sim, on="target", how="left").merge(
        feat[["building_id", "building_type"]].rename(columns={"building_id": "target"}),
        on="target", how="left",
    )
    feat.to_csv(out / "table_building_similarity_features.csv", index=False)
    merged.to_csv(out / f"table_{args.model}_similarity_transfer_gains.csv", index=False)

    for gain_col in [c for c in ["gain_m3_vs_m2", "gain_srft_vs_m3"] if c in merged.columns]:
        plot_df = merged.dropna(subset=["nearest_distance", gain_col])
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        for k, grp in plot_df.groupby("k"):
            ax.scatter(grp["nearest_distance"], grp[gain_col], s=28, label=f"k={k}", alpha=0.85)
        ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set_xlabel("Nearest source distance")
        ax.set_ylabel("MAE gain")
        ax.grid(axis="both", color="0.9", linewidth=0.7)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"figure_{args.model}_{gain_col}_vs_similarity.pdf")
        fig.savefig(out / f"figure_{args.model}_{gain_col}_vs_similarity.png", dpi=300)
        plt.close(fig)
    print(f"[similarity] saved to {out}")


if __name__ == "__main__":
    main()
