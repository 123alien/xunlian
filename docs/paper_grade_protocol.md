# Paper-Grade Experiment Protocol

This protocol defines the results that are acceptable for the paper-level TL-TFAD evaluation.

## Scope

Use existing `results/phase1` and pre-fix `results/phase2` outputs only as pilot/debug evidence.
Do not cite them as final paper results because the Phase 2 protocol has been tightened.

## Required Main Run

Run Phase 2 with:

```powershell
python run_phase2.py --skip-preprocess --buildings 12 --k 0 1 3 7 14 --seeds 42 43 44 45 46 --models lstm dlinear
```

If runtime is too high, keep the main comparison at 12 buildings and 5 seeds for LSTM, then run
DLinear on the same buildings with at least 3 seeds and label it clearly.

## Validity Rules

- `k=0` scaling must use target training data, not target test data.
- M2 and M3 must be compared pairwise by target building, k value, model, and seed.
- Seeds are repeated runs, not independent scientific samples. Aggregate seeds within each building-k unit before statistical testing.
- Anomaly thresholds must not be tuned on injected test labels.
- Synthetic injection prevalence must be point-level controlled at the stated ratio.
- Baseline anomaly detectors must be evaluated with anomaly labels, not forecasting MAE/RMSE against zero predictions.

## Required Result Tables

- `table_few_shot_transfer.csv`: MAE, RMSE, sMAPE, sigma_err, sigma_score for M1/M2/M3.
- `table_anomaly_injection.csv`: Precision, Recall, F1, AUROC, AUPRC, FAR, and per-type recall for M2/M3.
- `table_baseline_comparison.csv`: Precision, Recall, F1, AUROC, AUPRC, FAR, and per-type recall for Isolation Forest, LOF, and LSTM-AE.
- `table_negative_transfer.csv`: cases where M3 is worse than M2 in MAE, sigma_err, sigma_score, AUPRC, or FAR.
- `table_statistical_tests.csv`: paired tests and effect sizes using building-k units.

## Minimum Claim Bar

The main claim is paper-ready only if M3 improves residual stability and anomaly-detection reliability
over M2 across most building-k units. Strong final evidence should show:

- lower sigma_err or sigma_score in at least 70% of building-k units;
- no systematic MAE/RMSE degradation;
- higher AUPRC or F1 and lower FAR under controlled injection;
- negative transfer cases are limited and explainable by building type or load-profile mismatch.

If these conditions are not met, reframe the paper around when transfer helps or fails in cold-start
building anomaly detection.
