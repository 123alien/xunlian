# Expanded Energy and Buildings Experiment Plan

## Main Claim

Cold-start building energy forecasting should be evaluated by target-data budget.
The expected paper claim is not that one method dominates everywhere, but that:

- target-only neural forecasting is unreliable under few-shot target data;
- cross-building pretraining makes neural models usable;
- DLinear-SRFT consistently improves standard fine-tuning;
- tree baselines are strongest under extremely small target data, while neural transfer becomes competitive as k increases.

## Expanded Evidence Matrix

### Dataset Scale

- Move from 12 buildings to 24 buildings when the full BDG2 subset is available.
- Use a saved `building_manifest_24.csv` instead of alphabetical selection.
- Limit over-representation by setting a maximum of 6 buildings per building type.

### Target Data Budgets

- Main: `k = 3, 7, 14, 30`.
- Optional diagnostic: `k = 1` only if the paper needs an extreme cold-start point.

### Forecasting Methods

- Neural target-only: LSTM, DLinear, PatchTST.
- Neural transfer: LSTM, DLinear, PatchTST with source pretraining plus target fine-tuning.
- Proposed method: DLinear-SRFT with source replay fine-tuning.
- Classical baselines: persistence, seasonal naive, Ridge, Random Forest, ExtraTrees, HistGBDT.

### Ablations

- DLinear-SRFT replay weight: `lambda_source = 0.05, 0.1, 0.2, 0.3`.
- Report whether SRFT improves over standard DLinear fine-tuning at building-level, not seed-level.

### Explanatory Analysis

- Compute source-target similarity using load mean, variance, coefficient of variation, daily profile range, weekly profile range, and floor area.
- Plot similarity versus transfer gain:
  - `M3_pretrain_ft` gain over `M2_target_only`;
  - DLinear-SRFT gain over DLinear M3.

## Statistical Unit

Aggregate seeds within each building-k-method before formal comparison.
Use building as the independent unit.

## Recommended Run Order

1. Generate balanced 24-building manifest.
2. Run neural matrix for DLinear, LSTM, PatchTST at `k=3 7 14 30`.
3. Run classical forecasting baselines on the same manifest and k values.
4. Run DLinear-SRFT with replay weights `0.05 0.1 0.2 0.3`.
5. Aggregate final forecasting tables.
6. Run source-target similarity analysis.
7. Regenerate Energy and Buildings style figures.

## Reviewer Risks This Plan Addresses

- Missing modern baseline: PatchTST is included.
- Cherry-picking buildings: manifest-based balanced selection is saved.
- Seed pseudoreplication: final tests use building-level aggregation.
- Weak mechanism explanation: similarity analysis tests when transfer helps.
- Overclaiming: results will explicitly report the k=3 tree-model advantage if it remains.
