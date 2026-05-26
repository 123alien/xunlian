# Experimental Angle: BDG2 and COFACTOR as Complementary Cold-Start Settings

## Purpose

This note reinterprets the current experiments from a paper-design perspective. The goal is not to force a single "PARBoost is universally best" narrative, but to use BDG2 and COFACTOR as complementary experimental settings that reveal when persistence-anchored residual transfer is valuable and where strong tree ensembles remain difficult to beat.

## Core Experimental Thesis

Cold-start building load forecasting has at least two regimes:

1. **Heterogeneous, persistence-dominated transfer regime**: buildings differ strongly in scale, activity, and residual dynamics. Direct transfer of absolute load patterns can fail badly. Persistence anchoring and residual normalization are crucial.
2. **Active-building, tree-friendly regime**: target buildings are active, source-target tabular similarity is informative, and strong source-target tree ensembles perform extremely well. PARBoost remains persistence-safe but may not dominate ExtraTrees-ST.

BDG2 primarily supports the first regime. COFACTOR primarily tests the second regime. The paper should use this contrast as a strength, not hide it.

## Dataset Roles

### BDG2 Main Benchmark, 24 Buildings

Role:

- Full method matrix.
- Neural transfer failure diagnosis.
- PARBoost main method evaluation.
- Ablation of direct load prediction vs residual anchoring.

Supported conclusion:

PARBoost ranks first by mean rank on the main benchmark, and residual anchoring is the main mechanism that prevents transfer failure.

Key results:

- PARBoost mean MAE:
  - k=3: 2.871
  - k=7: 2.627
  - k=14: 2.488
  - k=30: 2.164
- PARBoost mean rank:
  - k=3: 1.71
  - k=7: 1.83
  - k=14: 1.83
  - k=30: 1.42
- DLinear-SRFT has severe worst-case failures, especially k=3.
- Direct HistGBDT-ST ablation fails badly, with MAE around 17-19, whereas residual variants reduce MAE to roughly 2-3.

Interpretation:

BDG2 is where the mechanism is clearest: direct transfer of absolute loads is unsafe; persistence-anchored residual learning solves the dominant failure mode.

Important caveat:

Some BDG2 buildings have near-zero or low-activity test periods. This is not a bug, but it means absolute MAE and relative-to-persistence ratios must be interpreted carefully. Relative MAE can become unstable when persistence MAE is near zero.

### BDG2 Robustness Benchmark, 120 Active Buildings

Role:

- Addresses reviewer concern that the main benchmark is too small.
- Tests whether PARBoost remains strong on a much larger active subset.
- Uses only strong classical baselines and PARBoost, not the full neural matrix.

Supported conclusion:

PARBoost remains the best-ranked method across all k values on a larger 120-building active BDG2 subset.

Key results:

- PARBoost mean MAE:
  - k=3: 18.319
  - k=7: 17.439
  - k=14: 17.111
  - k=30: 16.688
- Persistence mean MAE: 19.484 across k.
- RF-ST mean MAE:
  - k=3: 22.354
  - k=7: 21.393
  - k=14: 20.958
  - k=30: 20.686
- PARBoost mean rank:
  - k=3: 1.81
  - k=7: 1.76
  - k=14: 1.73
  - k=30: 1.64
- PARBoost win rate vs Persistence:
  - k=3: 90.0%
  - k=7: 93.3%
  - k=14: 95.8%
  - k=30: 96.7%
- PARBoost win rate vs RF-ST:
  - k=3: 75.8%
  - k=7: 74.2%
  - k=14: 75.8%
  - k=30: 78.3%

Interpretation:

This is strong robustness evidence. The larger subset has larger loads and greater variability, so its absolute MAE is not directly comparable to the 24-building benchmark. Within its own protocol, however, PARBoost is consistently best by mean rank, mean MAE, and win rate.

### COFACTOR Full Active Validation, 44 Buildings

Role:

- External validation.
- Tests whether PARBoost generalizes outside BDG2.
- Tests performance in active-building conditions where tree ensembles are strong.

Supported conclusion:

PARBoost is persistence-safe and competitive with tree ensembles, but ExtraTrees-ST remains strongest on COFACTOR.

Key results:

- ExtraTrees-ST mean MAE:
  - k=3: 3.859
  - k=7: 3.858
  - k=14: 3.771
  - k=30: 3.706
- PARBoost mean MAE:
  - k=3: 3.891
  - k=7: 3.927
  - k=14: 3.832
  - k=30: 3.743
- Persistence mean MAE: 4.395 across k.
- PARBoost relative MAE vs Persistence:
  - k=3: 0.891
  - k=7: 0.901
  - k=14: 0.876
  - k=30: 0.849
- PARBoost win rate vs Persistence:
  - k=3: 100.0%
  - k=7: 95.5%
  - k=14: 97.7%
  - k=30: 100.0%
- PARBoost win rate vs ExtraTrees-ST:
  - k=3: 34.1%
  - k=7: 31.8%
  - k=14: 29.5%
  - k=30: 36.4%
- PARBoost worst-case MAE is lower than ExtraTrees-ST at k=3, k=14, and k=30, but this is not enough to claim overall superiority.

Interpretation:

COFACTOR should not be presented as superiority evidence. Its value is boundary-setting: in active-building external validation, PARBoost reliably improves over persistence and is close to RF-ST/HistGBDT-ST, but ExtraTrees-ST remains the strongest method. This supports a nuanced conclusion: persistence-anchored residual transfer is robust, but strong tree ensembles remain indispensable baselines.

## What The Experiments Can Defensibly Claim

### Strong Claim 1

Direct transfer of absolute load dynamics is unsafe in cold-start building forecasting.

Evidence:

- Direct HistGBDT-ST ablation fails badly on BDG2.
- DLinear-SRFT and related neural transfer methods show large worst-case errors.
- Residual variants reduce MAE dramatically.

### Strong Claim 2

Persistence anchoring is the key mechanism behind PARBoost.

Evidence:

- Direct load prediction MAE is around 17-19 in the ablation.
- Residual-anchored variants reduce MAE to roughly 2-3 in the 24-building BDG2 ablation.
- PARBoost beats Persistence on 90-97% of buildings in BDG2-120, depending on k.

### Strong Claim 3

PARBoost is especially effective in heterogeneous BDG2 settings.

Evidence:

- PARBoost ranks first across all k values in BDG2-24.
- PARBoost also ranks first across all k values in BDG2-120.
- The advantage holds both in the full method benchmark and the larger active-building robustness check.

### Strong Claim 4

COFACTOR confirms robustness against persistence, not universal superiority.

Evidence:

- PARBoost beats Persistence on 95-100% of COFACTOR buildings.
- PARBoost relative MAE vs Persistence is 0.849-0.901.
- ExtraTrees-ST remains best by mean rank and mean MAE.

## Claims To Avoid

Do not claim:

- PARBoost is universally best across datasets.
- PARBoost replaces ExtraTrees-ST or RF-ST.
- COFACTOR proves superiority.
- Neural transfer is useless.
- The 120-building robustness MAE can be directly compared with the 24-building MAE.
- Seeds are independent samples.

## Better Experimental Framing

The paper should be organized around **mechanism-specific experiments**, not a single leaderboard:

### Experiment 1: Failure Mechanism

Question:

Why does direct transfer fail?

Evidence:

- Direct HistGBDT-ST vs residual variants.
- DLinear/SRFT worst-case failures.
- Persistence-dominated and low-residual cases.

Main message:

Cold-start transfer should not predict absolute load directly across heterogeneous buildings.

### Experiment 2: Method Effectiveness On Main Benchmark

Question:

Does PARBoost solve the identified BDG2 failure mode?

Evidence:

- BDG2-24 ranking.
- Mean rank, top-3 rate, worst-case MAE.
- Paired deltas vs Persistence, RF-ST, ExtraTrees-ST.

Main message:

PARBoost is the best method on the main BDG2 cold-start benchmark.

### Experiment 3: Large-Scale BDG2 Robustness

Question:

Is the BDG2 result an artifact of 24 selected buildings?

Evidence:

- BDG2-120 active robustness subset.
- PARBoost rank and win rate.

Main message:

The BDG2 advantage persists on a much larger active subset.

### Experiment 4: External Active-Building Boundary

Question:

Does the method generalize to a different active-building dataset?

Evidence:

- COFACTOR-44.
- Relative MAE vs Persistence.
- Win rate vs Persistence.
- Comparison against ExtraTrees-ST/RF-ST.

Main message:

PARBoost remains persistence-safe and competitive, but ExtraTrees-ST is strongest in this external active-building regime.

## Best Results Narrative

The strongest narrative is:

1. Cold-start transfer fails when models transfer absolute load dynamics instead of target-specific residual dynamics.
2. Persistence anchoring converts the problem into residual correction, which directly targets this failure.
3. In BDG2, where heterogeneity and persistence-dominated regimes are pronounced, PARBoost consistently outperforms strong baselines.
4. In the larger BDG2 active subset, this advantage persists across 120 buildings.
5. In COFACTOR, an external active-building dataset, PARBoost remains safer than Persistence and neural transfer but does not beat ExtraTrees-ST, defining the method's boundary.

This is more convincing than saying "PARBoost is close to ExtraTrees on COFACTOR."

## Recommended Main-Paper Placement

Main paper:

- BDG2-24 leaderboard.
- Ablation table.
- BDG2-120 robustness summary.
- COFACTOR-44 external validation summary.
- Failure-mechanism figure.
- Paired delta / win-rate figure.

Supplement:

- Full per-building tables.
- Full eligibility audit.
- All k/seed raw summaries.
- COFACTOR pairwise details.
- Additional neural baselines.

## Generated Evidence Tables

The paper-facing evidence tables are generated by:

```text
python scripts/generate_parboost_evidence_tables.py
```

Outputs:

- `results/parboost_evidence/table_tail_risk_by_dataset_method.csv`
- `results/parboost_evidence/table_paired_tests_parboost.csv`
- `results/parboost_evidence/table_stratified_by_persistence_strength.csv`
- `results/parboost_evidence/table_stratified_by_dataset_audit_features.csv`
- `results/parboost_evidence/table_all_validation_building_level_with_metadata.csv`

These tables should be used as the source for the Results section. The main manuscript should report building-paired comparisons, not seed-level comparisons, because buildings are the unit of generalization.

## Remaining Experimental Improvements

The most useful remaining analyses are not more model runs. They are:

1. Tail-risk summary:
   - P90/P95/worst MAE by method and k.
2. Failure-mode stratification:
   - group by test persistence MAE, test volatility, zero ratio, and building type.
3. Statistical testing:
   - Wilcoxon signed-rank tests and bootstrap confidence intervals over buildings.
4. Mechanism plots:
   - direct vs residual ablation.
   - PARBoost gain vs persistence MAE.
   - PARBoost gain vs source-target tree baselines.
