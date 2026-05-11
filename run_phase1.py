#!/usr/bin/env python
"""Run Phase 1 core claim validation.

Usage:
    python run_phase1.py [--buildings 5] [--k-days 7]

This script:
1. Downloads BDG2 if needed
2. Preprocesses the data
3. Runs Phase 1 core claim experiment
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocess import run_preprocessing_pipeline
from src.phase1 import run_phase1


def main():
    parser = argparse.ArgumentParser(description="TL-TFAD Phase 1: Core claim validation")
    parser.add_argument("--buildings", type=int, default=5,
                        help="Number of buildings to use (default: 5)")
    parser.add_argument("--k-days", type=int, default=7,
                        help="Few-shot data in days (default: 7)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip data download and preprocessing")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/processed/bdg2_electricity_hourly.csv")
    parser.add_argument("--output", default="results/phase1")
    args = parser.parse_args()

    # Step 1: Preprocess data
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
            print(f"[main] Processed data already exists: {args.data}")
    else:
        print("[main] Skipping preprocessing (--skip-preprocess)")

    # Step 2: Run Phase 1 experiment
    print("\n" + "=" * 60)
    print("STEP 2: Phase 1 core claim experiment")
    print("=" * 60)

    # Override k_days in the phase1 module
    import src.phase1 as p1
    p1.K_DAYS = args.k_days
    p1.K_HOURS = args.k_days * 24

    run_phase1(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        n_buildings=args.buildings,
    )


if __name__ == "__main__":
    main()
