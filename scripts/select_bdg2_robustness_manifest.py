#!/usr/bin/env python
"""Select a larger stratified robustness subset from a BDG2 eligibility audit."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/bdg2_full_eligible/table_eligibility_audit.csv")
    ap.add_argument("--output", default="results/bdg2_full_eligible/manifest_active_robustness_120.csv")
    ap.add_argument("--n-buildings", type=int, default=120)
    ap.add_argument("--max-per-type", type=int, default=12)
    args = ap.parse_args()

    audit = pd.read_csv(args.audit)
    eligible = audit[audit["selected"]].copy()
    if eligible.empty:
        raise SystemExit("No selected buildings found in audit.")

    eligible["type_count"] = eligible.groupby("building_type")["building_id"].transform("count")
    buckets = {
        typ: grp.sort_values(["test_persistence_mae", "test_std", "building_id"], ascending=[False, False, True]).to_dict("records")
        for typ, grp in eligible.groupby("building_type", sort=True)
    }
    selected = []
    per_type = {typ: 0 for typ in buckets}
    while len(selected) < args.n_buildings:
        progressed = False
        for typ in sorted(buckets, key=lambda t: (per_type[t], -len(buckets[t]), t)):
            if len(selected) >= args.n_buildings:
                break
            if per_type[typ] >= args.max_per_type or not buckets[typ]:
                continue
            selected.append(buckets[typ].pop(0))
            per_type[typ] += 1
            progressed = True
        if not progressed:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(selected)
    df.to_csv(out, index=False)
    print(f"[bdg2-robustness-manifest] eligible_active={len(eligible)} selected={len(df)} saved={out}")
    print(df.groupby("building_type")["building_id"].count().sort_values(ascending=False).to_string())
    print(df[["building_id", "building_type", "test_persistence_mae", "test_std", "test_zero_ratio"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
