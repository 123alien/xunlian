# Forecasting Paper Experiment Design

## Paper Positioning

The paper is repositioned as a cold-start building energy forecasting study:

> Cross-building transfer learning improves few-shot hourly electricity forecasting when only a few days of target-building data are available.

Anomaly detection is no longer the central claim. It is treated as a downstream motivation and a limitation-driven discussion point.

## Research Questions

**RQ1.** Does source-building pretraining plus target fine-tuning outperform target-only few-shot training?

- Central comparison: M3 pretrain+fine-tune vs M2 target-only.
- Main evidence: MAE, RMSE, sMAPE, sigma_err.
- Main statistical unit: target building, after averaging repeated seeds.

**RQ2.** Are improvements strongest in colder-start settings?

- Compare k = 3, 7, 14 days.
- Expected pattern: larger gains at k = 3, smaller gains at k = 14.

**RQ3.** Are gains robust across model families?

- Current models: LSTM and DLinear.
- Required interpretation: if both architectures improve, the result supports the transfer protocol rather than one model implementation.

**RQ4.** Does the proposed transfer protocol beat simple and classical forecasting baselines?

- Baselines:
  - Persistence: next hour equals last observed hour.
  - Seasonal naive: next hour equals the same hour from the previous day.
  - Ridge regression.
  - Random Forest.
  - Extra Trees.
  - Histogram Gradient Boosting.
- Classical ML baselines should be evaluated in both target-only and source+target pooled modes when applicable.

## Data Protocol

- Dataset: BDG2 hourly electricity data.
- Target buildings: 12 buildings used in the completed Phase 2 experiments.
- Source-target split: leave-one-building-out.
- Time split within each target building:
  - first 60%: training pool,
  - next 20%: validation,
  - last 20%: test.
- Few-shot target data:
  - k = 3, 7, 14 days selected only from the beginning of the target training pool.
- Test set:
  - never used for scaling, model selection, or hyperparameter selection.

## Scaling And Leakage Rules

- For target-only baselines, scalers are fit only on k-day target training windows.
- For source+target pooled baselines, scalers are fit only on source training windows plus k-day target training windows.
- For k=0 or source-only context, target test distribution must not be used for scaling.
- Validation and test data are transformed using training-fitted scalers only.

## Methods To Compare

### Existing Core Methods

- M2 target-only LSTM.
- M3 source-pretrained + target-finetuned LSTM.
- M2 target-only DLinear.
- M3 source-pretrained + target-finetuned DLinear.

### New Forecasting Baselines

Zero-training:

- Persistence.
- Seasonal naive.

Classical supervised:

- Ridge target-only.
- Ridge source+target pooled.
- Random Forest target-only.
- Random Forest source+target pooled.
- Extra Trees target-only.
- Extra Trees source+target pooled.
- HistGradientBoosting target-only.
- HistGradientBoosting source+target pooled.

These baselines are intentionally chosen because they are strong enough to be credible, easy to reproduce, and available through scikit-learn.

## Metrics

Main forecasting metrics:

- MAE.
- RMSE.
- sMAPE.

Stability metric:

- sigma_err: standard deviation of absolute residuals.

Secondary/exploratory:

- sigma_score should not be used as a core positive claim because previous results are inconsistent.

## Statistical Testing

Main-text statistics:

1. Compute metric values for each target building and seed.
2. Average repeated seeds within each target building.
3. Run paired tests over target buildings (`n=12`) for M3 - M2.
4. Report:
   - mean delta,
   - bootstrap 95% CI,
   - Wilcoxon signed-rank p-value,
   - Cliff's delta against zero,
   - building-level win rate.

Seed-level `n=60` tables can be placed in supplementary material only.

## Main Figures And Tables

Main Table 1:

- Dataset and split summary, including building types and number of samples.

Main Table 2:

- Building-level M3 vs M2 deltas for MAE and sigma_err across model/k settings.

Main Figure 1:

- MAE and sigma_err curves over k = 3, 7, 14 for M2 vs M3.

Main Figure 2:

- Building-level MAE deltas at k = 3, showing consistency and any negative/weak cases.

Supplementary Table:

- Classical baseline comparison.

Supplementary Figure:

- Per-building or per-building-type subgroup analysis.

## Reviewer Risks And Defenses

Risk: "This is just another forecasting paper."

Defense: The paper is not about a new neural architecture; it is an empirical cold-start transfer study with building-level statistics across multiple target buildings, architectures, and data budgets.

Risk: "Seeds are treated as independent samples."

Defense: Main statistics use building-level aggregation. Seed-level results are supplementary.

Risk: "Baselines are weak."

Defense: Add naive, linear, tree ensemble, and boosting baselines. Clearly separate target-only and source+target settings.

Risk: "Anomaly detection claim is unsupported."

Defense: Do not make anomaly detection the core claim. Discuss it only as an exploratory downstream application.

## Go/No-Go Criteria

The forecasting paper is ready for manuscript drafting if:

- M3 remains significantly better than M2 at building level for MAE and sigma_err.
- M3 is competitive with or better than classical source+target baselines in the lowest-data settings.
- Negative transfer cases are reported explicitly rather than hidden.

