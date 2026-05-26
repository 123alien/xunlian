#!/usr/bin/env python
"""Generate manuscript-ready PARBoost tables from existing result summaries."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "parboost_evidence" / "manuscript_tables"


METHOD_LABEL = {
    "PAR_HIST_GBDT_SW_CAL": "PARBoost",
    "PAR_EXTRA_TREES_SW_CAL": "PARBoost-ExtraTrees",
    "persistence": "Persistence",
    "seasonal_naive_24": "Seasonal naive",
    "random_forest_source_target": "RF-ST",
    "extra_trees_source_target": "ExtraTrees-ST",
    "hist_gbdt_source_target": "HistGBDT-ST",
    "dlinear_M3R_lambda0.3": "DLinear-SRFT",
    "dlinear_M3_pretrain_ft": "DLinear-M3",
    "Residual_M3": "Residual-M3",
    "Residual_SRFT": "Residual-SRFT",
    "PA_SRFT": "PA-SRFT",
    "ABL_DIRECT_HISTGBDT_ST": "Direct HistGBDT-ST",
    "ABL_RESIDUAL_UNIFORM": "Residual",
    "ABL_RESIDUAL_SIM": "Residual + similarity",
    "ABL_RESIDUAL_CAL": "Residual + calibration",
}


def save_table(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"{name}.csv"
    md_path = OUT / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df), encoding="utf-8")
    return csv_path, md_path


def to_markdown(df: pd.DataFrame) -> str:
    text_df = df.astype(str)
    headers = list(text_df.columns)
    rows = text_df.values.tolist()
    widths = []
    for j, h in enumerate(headers):
        widths.append(max(len(h), *(len(row[j]) for row in rows)) if rows else len(h))
    header = "| " + " | ".join(h.ljust(widths[j]) for j, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if "p" in col.lower():
                continue
            if col in {"k", "n", "n_buildings", "seeds", "target_buildings"}:
                continue
            out[col] = out[col].round(3)
    return out


def table_1_dataset_roles():
    rows = [
        {
            "dataset": "BDG2-24",
            "target_buildings": 24,
            "source_policy": "leave target building out",
            "split": "60/20/20 chronological",
            "k_days": "3, 7, 14, 30",
            "seeds": 5,
            "main_role": "full method matrix; neural-transfer failure and PARBoost main benchmark",
        },
        {
            "dataset": "BDG2-120 active",
            "target_buildings": 120,
            "source_policy": "leave target building out",
            "split": "60/20/20 chronological",
            "k_days": "3, 7, 14, 30",
            "seeds": 5,
            "main_role": "large active-building robustness against strong simple/classical baselines",
        },
        {
            "dataset": "COFACTOR-44 active",
            "target_buildings": 44,
            "source_policy": "leave target building out",
            "split": "60/20/20 chronological",
            "k_days": "3, 7, 14, 30",
            "seeds": 5,
            "main_role": "external active-building boundary validation",
        },
    ]
    return pd.DataFrame(rows)


def table_2_bdg2_24_full():
    df = pd.read_csv(ROOT / "results" / "parboost_expanded_bdg2" / "table_par_combined_ranking.csv")
    keep = [
        "PAR_HIST_GBDT_SW_CAL",
        "persistence",
        "random_forest_source_target",
        "extra_trees_source_target",
        "hist_gbdt_source_target",
        "dlinear_M3R_lambda0.3",
        "dlinear_M3_pretrain_ft",
        "PA_SRFT",
    ]
    df = df[df["method"].isin(keep)].copy()
    df["method"] = df["method"].map(METHOD_LABEL)
    cols = ["method", "k", "mean_MAE", "median_MAE", "worst_MAE", "mean_rank", "top3_rate"]
    return round_numeric(df[cols].sort_values(["k", "mean_rank", "mean_MAE"]))


def table_from_tail(dataset: str, methods: list[str]):
    df = pd.read_csv(ROOT / "results" / "parboost_evidence" / "table_tail_risk_by_dataset_method.csv")
    df = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].copy()
    df["method"] = df["method"].map(METHOD_LABEL)
    cols = [
        "method",
        "k",
        "mean_MAE",
        "median_MAE",
        "p90_MAE",
        "p95_MAE",
        "worst_MAE",
        "mean_rank",
        "top3_rate",
    ]
    return round_numeric(df[cols].sort_values(["k", "mean_rank", "mean_MAE"]))


def table_5_pairwise():
    df = pd.read_csv(ROOT / "results" / "parboost_evidence" / "table_paired_tests_parboost.csv")
    df = df[df["dataset"].isin(["BDG2-120", "COFACTOR-44"])].copy()
    df["candidate"] = df["candidate"].map(METHOD_LABEL)
    df["baseline"] = df["baseline"].map(METHOD_LABEL)
    cols = [
        "dataset",
        "baseline",
        "k",
        "n",
        "mean_delta",
        "mean_delta_ci95_low",
        "mean_delta_ci95_high",
        "median_delta",
        "win_rate",
        "loss_rate",
        "wilcoxon_p",
    ]
    out = round_numeric(df[cols].sort_values(["dataset", "baseline", "k"]))
    out["wilcoxon_p"] = out["wilcoxon_p"].map(lambda x: f"{float(x):.2e}" if pd.notna(x) else "")
    return out


def table_6_ablation():
    df = pd.read_csv(ROOT / "results" / "parboost_ablation_bdg2" / "table_par_gbdt_pilot.csv")
    keep = [
        "ABL_DIRECT_HISTGBDT_ST",
        "ABL_RESIDUAL_UNIFORM",
        "ABL_RESIDUAL_SIM",
        "ABL_RESIDUAL_CAL",
        "PAR_HIST_GBDT_SW_CAL",
    ]
    df = df[df["method"].isin(keep)].copy()
    building = df.groupby(["target", "k", "method"], as_index=False)["MAE"].mean()
    summary = building.groupby(["method", "k"], as_index=False).agg(
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        p90_MAE=("MAE", lambda s: s.quantile(0.90)),
        worst_MAE=("MAE", "max"),
    )
    summary["method"] = summary["method"].map(METHOD_LABEL)
    return round_numeric(summary.sort_values(["k", "mean_MAE"]))


def main():
    outputs = {
        "table1_dataset_roles": table_1_dataset_roles(),
        "table2_bdg2_24_full_leaderboard": table_2_bdg2_24_full(),
        "table3_bdg2_120_robustness": table_from_tail(
            "BDG2-120",
            [
                "PAR_HIST_GBDT_SW_CAL",
                "persistence",
                "random_forest_source_target",
                "extra_trees_source_target",
                "hist_gbdt_source_target",
                "seasonal_naive_24",
            ],
        ),
        "table4_cofactor_44_external": table_from_tail(
            "COFACTOR-44",
            [
                "PAR_HIST_GBDT_SW_CAL",
                "persistence",
                "random_forest_source_target",
                "extra_trees_source_target",
                "hist_gbdt_source_target",
                "seasonal_naive_24",
            ],
        ),
        "table5_pairwise_tests": table_5_pairwise(),
        "table6_parboost_ablation": table_6_ablation(),
    }
    for name, table in outputs.items():
        csv_path, md_path = save_table(table, name)
        print(f"saved {csv_path}")
        print(f"saved {md_path}")


if __name__ == "__main__":
    main()
