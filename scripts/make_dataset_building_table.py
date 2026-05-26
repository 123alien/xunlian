#!/usr/bin/env python
"""Create the 12-building metadata table for the forecasting manuscript."""
from pathlib import Path

import pandas as pd


def write_markdown(df: pd.DataFrame, path: Path):
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    data_path = Path("data/processed/bdg2_electricity_hourly.csv")
    result_path = Path("results/phase2_paper/aggregate/table_few_shot_transfer_all_models.csv")
    out_dir = Path("results/phase2_paper/final_forecasting_package")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path, parse_dates=["timestamp"])
    few = pd.read_csv(result_path)
    buildings = sorted(few["target"].unique())

    rows = []
    for building_id in buildings:
        g = data[data["building_id"] == building_id].copy()
        rows.append({
            "building_id": building_id,
            "building_type": g["building_type"].mode().iloc[0] if len(g) else "",
            "sqm": round(float(g["sqm"].dropna().iloc[0]), 1) if len(g) and g["sqm"].notna().any() else None,
            "n_hours": int(len(g)),
            "start": g["timestamp"].min().strftime("%Y-%m-%d") if len(g) else "",
            "end": g["timestamp"].max().strftime("%Y-%m-%d") if len(g) else "",
            "mean_energy": round(float(g["energy"].mean()), 3) if len(g) else None,
            "std_energy": round(float(g["energy"].std()), 3) if len(g) else None,
            "missing_energy": int(g["energy"].isna().sum()) if len(g) else None,
        })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_dataset_buildings.csv", index=False)
    write_markdown(table, out_dir / "table_dataset_buildings.md")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
