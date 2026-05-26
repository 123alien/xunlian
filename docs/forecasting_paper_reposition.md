# Forecasting-Focused Paper Reposition

## Recommended Title

Cross-building transfer learning for few-shot building energy load forecasting under cold-start conditions

## Central Claim

Cross-building pretraining followed by few-shot target adaptation consistently improves building-level load forecasting accuracy and residual stability when only a few days of target-building data are available.

## What The Current Results Support

- M3 (pretrain + fine-tune) is consistently better than M2 (target-only few-shot training) for MAE, RMSE, sMAPE, and residual error variability.
- The effect holds across two forecasting architectures: LSTM and DLinear.
- The effect holds across 12 target buildings and three few-shot settings: k=3, k=7, and k=14 days.
- Building-level statistical tests remain significant after averaging repeated seeds within each target building.

## Main Evidence To Use

Use `results/phase2_paper/forecasting_paper/table_building_level_claim_summary.csv` for main-text statistics.

Key building-level findings:

- DLinear k=3: mean delta MAE = -33.08, MAE win rate = 100%, Wilcoxon p = 0.0022.
- DLinear k=7: mean delta MAE = -12.86, MAE win rate = 100%, Wilcoxon p = 0.0022.
- DLinear k=14: mean delta MAE = -3.06, MAE win rate = 100%, Wilcoxon p = 0.0022.
- LSTM k=3: mean delta MAE = -14.33, MAE win rate = 100%, Wilcoxon p = 0.0022.
- LSTM k=7: mean delta MAE = -9.52, MAE win rate = 100%, Wilcoxon p = 0.0022.
- LSTM k=14: mean delta MAE = -3.68, MAE win rate = 100%, Wilcoxon p = 0.0022.

Residual stability (`sigma_err`) shows the same direction across almost all conditions and remains significant at the building level.

## Claims To Avoid

- Do not claim that transfer learning universally improves anomaly detection.
- Do not claim that lower prediction error necessarily yields better anomaly detection.
- Do not use sigma_score as a primary positive result; it is inconsistent and often worsens.
- Do not report seed-level n=60 as the primary statistical sample. Seeds are repeated runs; the main text should use building-level n=12.

## Role Of Anomaly Detection

Anomaly detection should be presented as a downstream motivation or exploratory extension, not as the central contribution.

Recommended wording:

> Improved residual stability is relevant to residual-based anomaly screening, but controlled injected-anomaly experiments show that detection performance also depends on scoring and calibration. Therefore, anomaly detection is treated as a downstream application rather than the primary claim of this study.

## Proposed Contributions

1. A cold-start cross-building forecasting protocol that evaluates whether source-building pretraining helps when the target building has only a few days of labeled energy data.
2. A paired empirical comparison of target-only training and pretrain-then-fine-tune adaptation across 12 buildings, two model families, three few-shot budgets, and five random seeds.
3. Building-level statistical evidence showing consistent improvements in forecasting accuracy and residual stability after aggregating repeated seeds within each target building.
4. An exploratory analysis showing that improved residual quality is a useful but insufficient condition for downstream anomaly detection, motivating calibrated event-level detectors as future work.

## Paper Structure

1. Introduction
   - Building energy forecasting is important for operational monitoring and control.
   - New or poorly instrumented buildings often lack enough target-specific data.
   - Cross-building transfer is promising, but its reliability under few-shot target adaptation needs systematic evaluation.
   - This paper evaluates when pretraining beats target-only learning under controlled cold-start conditions.

2. Related Work
   - Building energy load forecasting.
   - Transfer learning and domain adaptation for building energy modeling.
   - Few-shot and cold-start forecasting.
   - Forecasting residuals for monitoring and anomaly screening.

3. Methods
   - Problem formulation.
   - Leave-one-building-out source-target setup.
   - M2 target-only few-shot baseline.
   - M3 source pretraining plus target fine-tuning.
   - LSTM and DLinear implementations.
   - Chronological split, scaling policy, and reproducibility protocol.

4. Experiments
   - Dataset: BDG2 hourly electricity data.
   - Targets: 12 buildings across building types.
   - k-day few-shot settings: 3, 7, 14.
   - Metrics: MAE, RMSE, sMAPE, sigma_err.
   - Statistics: building-level paired Wilcoxon, bootstrap CI, Cliff's delta.

5. Results
   - M3 improves few-shot forecasting accuracy.
   - Gains are largest under the smallest target-data budgets.
   - Residual error variability is reduced, indicating more stable forecasting behavior.
   - Negative or weaker cases should be reported explicitly.

6. Discussion
   - Why cross-building pretraining helps.
   - Why gains shrink as k increases.
   - Limits of using forecasting residuals for anomaly detection.
   - Practical implications for newly deployed buildings.

7. Limitations
   - Synthetic or controlled anomaly experiments are exploratory.
   - Only two forecasting architectures are currently included.
   - Building coverage is stronger than the pilot but still not a full BDG2-scale benchmark.
   - Real fault labels are not available.

## Experiments Still Worth Adding

Highest priority:

- Seasonal naive or persistence baseline.
- Tree-based baseline such as XGBoost or Random Forest.
- One additional neural forecasting model, preferably GRU, TCN, N-BEATS, or PatchTST.

Medium priority:

- Per-building negative transfer table.
- Building-type subgroup analysis.
- Source-building count ablation.
- k=1 with a shorter input window, clearly marked as a supplementary ultra-cold-start setting.

Low priority:

- More anomaly detection optimization. Keep this out of the main paper unless detection metrics become consistently strong.

