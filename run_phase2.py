#!/usr/bin/env python
"""Run Phase 2 full experiment matrix.

Usage:
    # Minimal sanity check (3 buildings: 1 target + 2 sources)
    python run_phase2.py --sanity

    # Quick test (3 buildings, k=7, LSTM, 1 seed)
    python run_phase2.py --quick

    # Paper-grade Phase 2
    python run_phase2.py --buildings 12 --seeds 42 43 44 45 46

    # Partial run
    python run_phase2.py --k 1 3 7 --models lstm
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocess import run_preprocessing_pipeline
from src.phase2 import run_phase2


def main():
    parser = argparse.ArgumentParser(description="TL-TFAD Phase 2: Full experiment")
    parser.add_argument("--buildings", type=int, default=12,
                        help="Number of buildings (default: 12 for paper-grade runs)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 3 buildings, k=7, seed=42, LSTM only")
    parser.add_argument("--sanity", action="store_true",
                        help="Minimal sanity: 3 buildings, k=7, seed=42, LSTM")
    parser.add_argument("--core-claim", action="store_true",
                        help="Generate core_claim_check.csv/.md after run")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip AD baseline comparison")
    parser.add_argument("--skip-injection", action="store_true",
                        help="Skip anomaly injection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    parser.add_argument("--output", default="results/phase2")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="k values (days), e.g. --k 0 1 3 7 14")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="seeds, e.g. --seeds 42 43 44")
    parser.add_argument("--models", nargs="+", default=None,
                        help="models, e.g. --models lstm dlinear")
    args = parser.parse_args()

    # Preprocess if needed
    if not args.skip_preprocess:
        processed_path = Path(args.data)
        if not processed_path.exists():
            print("=" * 60)
            print("STEP 1: Data preprocessing")
            print("=" * 60)
            run_preprocessing_pipeline(
                raw_dir="data/raw",
                processed_dir="data/processed",
                n_buildings=args.buildings,
            )
        else:
            print(f"[main] Using existing data: {args.data}")

    # Run Phase 2
    print("\n" + "=" * 60)
    print("STEP 2: Phase 2 experiment matrix")
    print("=" * 60)

    if args.quick:
        run_phase2(
            config_path=args.config, data_path=args.data, output_dir=args.output,
            n_buildings=3, k_values=[7], seeds=[42], models=["lstm"],
            skip_baselines=True, skip_injection=True,
        )
    elif args.sanity:
        run_phase2(
            config_path=args.config, data_path=args.data, output_dir=args.output,
            n_buildings=3, k_values=[7], seeds=[42], models=["lstm"],
            skip_baselines=True, skip_injection=True,
        )
    else:
        n_bldgs = args.buildings
        run_phase2(
            config_path=args.config, data_path=args.data, output_dir=args.output,
            n_buildings=n_bldgs,
            k_values=args.k, seeds=args.seeds, models=args.models,
            skip_baselines=args.skip_baselines,
            skip_injection=args.skip_injection,
        )

    if args.core_claim or args.sanity or args.quick:
        generate_core_claim_check(args.output)


def generate_core_claim_check(output_dir: str = "results/phase2"):
    """Generate core_claim_check.csv and core_claim_check.md from Phase 2 results."""
    import json

    out_root = Path(output_dir)
    rows = []
    for metrics_file in out_root.rglob("metrics.json"):
        with open(metrics_file) as f:
            m = json.load(f)
        if m.get("k") != 7 or m.get("method", "") not in ("M2_target_only", "M3_pretrain_ft"):
            continue
        rows.append(m)

    if not rows:
        print("[core_claim] No k=7 M2/M3 results found. Skipping report.")
        return

    agg_dir = out_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    buildings = sorted(set(r["target"] for r in rows))
    table_rows = []
    for bid in buildings:
        m2_rows = [r for r in rows if r["target"] == bid and r["method"] == "M2_target_only"]
        m3_rows = [r for r in rows if r["target"] == bid and r["method"] == "M3_pretrain_ft"]
        if not m2_rows or not m3_rows:
            continue
        m2 = m2_rows[0]
        m3 = m3_rows[0]
        sigma_err_m2 = m2.get("sigma_err", float("nan"))
        sigma_err_m3 = m3.get("sigma_err", float("nan"))
        sigma_score_m2 = m2.get("sigma_score", float("nan"))
        sigma_score_m3 = m3.get("sigma_score", float("nan"))
        mae_m2 = m2.get("MAE", m2.get("mae", float("nan")))
        mae_m3 = m3.get("MAE", m3.get("mae", float("nan")))
        rmse_m2 = m2.get("RMSE", m2.get("rmse", float("nan")))
        rmse_m3 = m3.get("RMSE", m3.get("rmse", float("nan")))

        d_err = sigma_err_m3 - sigma_err_m2
        d_score = sigma_score_m3 - sigma_score_m2
        d_mae = mae_m3 - mae_m2
        d_rmse = rmse_m3 - rmse_m2

        table_rows.append({
            "building": bid,
            "sigma_err_M2": sigma_err_m2, "sigma_err_M3": sigma_err_m3, "delta_sigma_err": d_err,
            "sigma_score_M2": sigma_score_m2, "sigma_score_M3": sigma_score_m3, "delta_sigma_score": d_score,
            "MAE_M2": mae_m2, "MAE_M3": mae_m3, "delta_MAE": d_mae,
            "RMSE_M2": rmse_m2, "RMSE_M3": rmse_m3, "delta_RMSE": d_rmse,
            "sigma_err_win": "M3" if d_err < 0 else "M2",
            "sigma_score_win": "M3" if d_score < 0 else "M2",
        })

    import pandas as pd
    df = pd.DataFrame(table_rows)
    csv_path = agg_dir / "core_claim_check.csv"
    df.to_csv(csv_path, index=False)
    print(f"[core_claim] Saved {csv_path}")

    n_win_err = sum(1 for r in table_rows if r["sigma_err_win"] == "M3")
    n_win_score = sum(1 for r in table_rows if r["sigma_score_win"] == "M3")
    n_total = len(table_rows)

    lines = [
        "# Phase 2 Core Claim Validation Report",
        "",
        f"**Date**: 2026-05-11",
        f"**Seed**: 42",
        f"**k**: 7 days",
        f"**Buildings**: {n_total} ({', '.join(buildings)})",
        f"**Model**: LSTM (hidden=128, layers=2)",
        "",
        "## Core Claim",
        "",
        "> Pretrain + fine-tune (M3) produces **lower sigma_err and sigma_score**",
        "> than target-only few-shot (M2), indicating more stable prediction",
        "> errors and more reliable anomaly scoring.",
        "",
        "## Results",
        "",
        "| Building | sigma_err M2 | sigma_err M3 | Delta | sigma_score M2 | sigma_score M3 | Delta | MAE M2 | MAE M3 | RMSE M2 | RMSE M3 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['building']} "
            f"| {r['sigma_err_M2']:.4f} | {r['sigma_err_M3']:.4f} | {r['delta_sigma_err']:+.4f} "
            f"| {r['sigma_score_M2']:.4f} | {r['sigma_score_M3']:.4f} | {r['delta_sigma_score']:+.4f} "
            f"| {r['MAE_M2']:.4f} | {r['MAE_M3']:.4f} "
            f"| {r['RMSE_M2']:.4f} | {r['RMSE_M3']:.4f} |"
        )

    mean_d_err = sum(r["delta_sigma_err"] for r in table_rows) / n_total
    mean_d_score = sum(r["delta_sigma_score"] for r in table_rows) / n_total
    mae_degrade = sum(1 for r in table_rows if r["delta_MAE"] > 0)
    rmse_degrade = sum(1 for r in table_rows if r["delta_RMSE"] > 0)

    lines += [
        "",
        "## Summary",
        "",
        f"- **sigma_err**: M3 wins in {n_win_err}/{n_total} buildings (mean Delta = {mean_d_err:.4f})",
        f"- **sigma_score**: M3 wins in {n_win_score}/{n_total} buildings (mean Delta = {mean_d_score:.4f})",
        f"- **MAE degradation**: {mae_degrade}/{n_total} buildings",
        f"- **RMSE degradation**: {rmse_degrade}/{n_total} buildings",
        "",
        "## Core Claim Questions",
        "",
    ]

    if n_win_err >= n_total * 0.5:
        lines.append(f"1. **M3 sigma_err < M2 sigma_err?** YES — M3 wins in {n_win_err}/{n_total} buildings.")
    else:
        lines.append(f"1. **M3 sigma_err < M2 sigma_err?** NO — M3 wins in only {n_win_err}/{n_total} buildings.")

    if n_win_score >= n_total * 0.5:
        lines.append(f"2. **M3 sigma_score < M2 sigma_score?** YES — M3 wins in {n_win_score}/{n_total} buildings.")
    else:
        lines.append(f"2. **M3 sigma_score < M2 sigma_score?** NO — M3 wins in only {n_win_score}/{n_total} buildings.")

    if mae_degrade <= n_total * 0.5 and rmse_degrade <= n_total * 0.5:
        lines.append(f"3. **MAE/RMSE not significantly worse?** YES — degradation in {mae_degrade}/{rmse_degrade} buildings.")
    else:
        lines.append(f"3. **MAE/RMSE not significantly worse?** CONCERN — degradation in {mae_degrade}/{rmse_degrade} buildings.")

    if n_win_err >= n_total * 0.5 and n_win_score >= n_total * 0.5:
        lines.append(f"4. **Worth running full Phase 2?** YES — core claim supported, proceed to full matrix.")
    else:
        lines.append(f"4. **Worth running full Phase 2?** MIXED — investigate further before scaling up.")

    lines += [
        "",
        "## Next Steps",
        "",
        "Run full Phase 2: `python run_phase2.py --n-buildings 6`",
    ]

    md_path = agg_dir / "core_claim_check.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[core_claim] Saved {md_path}")


if __name__ == "__main__":
    main()
