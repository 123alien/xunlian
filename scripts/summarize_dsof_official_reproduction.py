from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def scalar_from_npy(path: Path) -> float:
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.size == 0:
        return float("nan")
    return float(np.ravel(arr)[-1])


def parse_result_path(path: Path) -> dict[str, str]:
    parts = path.parts
    method_dir = ""
    dataset_pred = ""
    timestamp = ""
    itr = ""
    for i, part in enumerate(parts):
        if part.startswith("DLinear--MLP--"):
            method_dir = part
        if re.match(r"^[A-Za-z0-9]+_pl\d+$", part):
            dataset_pred = part
            if i + 1 < len(parts):
                timestamp = parts[i + 1]
            if i + 2 < len(parts):
                itr = parts[i + 2]

    dataset, pred_len = "", ""
    if "_pl" in dataset_pred:
        dataset, pred_len = dataset_pred.rsplit("_pl", 1)

    method = method_dir.replace("DLinear--MLP--", "")
    method = method.replace("w_student_residual_", "")

    return {
        "method": method,
        "dataset": dataset,
        "pred_len": pred_len,
        "timestamp": timestamp,
        "itr": itr.replace("itr", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="external/iclr2025_dsof")
    parser.add_argument("--out", default="results/dsof_official_reproduction_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for metrics_path in sorted(root.glob("exps/**/results/metrics.npy")):
        result_dir = metrics_path.parent
        row = parse_result_path(metrics_path)
        row["path"] = str(result_dir.parent)
        row["mae"] = scalar_from_npy(result_dir / "mae.npy") if (result_dir / "mae.npy").exists() else float("nan")
        row["mse"] = scalar_from_npy(result_dir / "mse.npy") if (result_dir / "mse.npy").exists() else float("nan")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "pred_len", "method", "itr", "timestamp", "mae", "mse", "path"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
