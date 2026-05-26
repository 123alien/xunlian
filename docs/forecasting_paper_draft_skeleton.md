# Draft Skeleton: Forecasting-Focused Manuscript

## Working Title

Cross-building transfer learning for few-shot building energy load forecasting under cold-start conditions

## Abstract Draft

Accurate building energy load forecasting is essential for operational monitoring, control, and energy management, but newly deployed or sparsely monitored buildings often provide only a few days of target-specific data. This cold-start setting limits the reliability of models trained from scratch and motivates cross-building transfer learning. In this study, we evaluate whether source-building pretraining followed by few-shot target adaptation improves hourly electricity forecasting under limited target data. Using 12 buildings from the BDG2 dataset, we compare target-only few-shot training with pretrain-then-fine-tune adaptation across two forecasting architectures, LSTM and DLinear, and three target-data budgets of 3, 7, and 14 days. Experiments use chronological splits and building-level paired statistics, with repeated seeds averaged within each target building. Across all model and data-budget settings, transfer learning reduces MAE, RMSE, sMAPE, and residual error variability relative to target-only training. At the building level, MAE improvements are significant for both architectures and all few-shot budgets (Wilcoxon p = 0.0022), with the largest gains observed in the lowest-data setting. These results provide evidence that cross-building pretraining is a robust strategy for cold-start building load forecasting. Exploratory anomaly-screening experiments further indicate that improved residual quality is useful but insufficient by itself for anomaly detection, highlighting the need for calibrated detection layers in downstream monitoring tasks.

## Core Contributions

1. We formulate a cold-start cross-building forecasting protocol for evaluating whether source-building pretraining improves target-building prediction when only a few days of target data are available.
2. We compare target-only training and pretrain-then-fine-tune adaptation across 12 buildings, two model families, three few-shot budgets, and five random seeds.
3. We report building-level paired statistical tests that aggregate repeated seeds within each target building, avoiding seed-level pseudoreplication in the main evidence.
4. We show that forecasting residual stability improves consistently under transfer learning, while downstream anomaly detection requires additional calibrated scoring and should not be inferred from forecasting accuracy alone.

## Results Narrative Draft

### Transfer learning consistently improves few-shot forecasting

Across both forecasting architectures, pretraining on source buildings followed by target fine-tuning consistently outperformed target-only few-shot training. At the building level, M3 reduced MAE for every evaluated model and few-shot budget, with 100% building-level win rates for DLinear and LSTM at k = 3, 7, and 14 days. The magnitude of improvement was largest when target data were most limited. For DLinear, the mean MAE reduction was 33.08 at k = 3, 12.86 at k = 7, and 3.06 at k = 14. For LSTM, the corresponding reductions were 14.33, 9.52, and 3.68. Building-level Wilcoxon signed-rank tests were significant in all six settings (p = 0.0022), and bootstrap confidence intervals for mean MAE differences did not cross zero.

### Gains extend beyond MAE to residual stability

The same pattern was observed for RMSE, sMAPE, and residual error variability. Transfer learning reduced sigma_err across nearly all building-level comparisons, indicating that pretraining did not merely lower average errors but also made prediction residuals more stable. This residual stabilization is important for operational monitoring because highly volatile residuals can make threshold-based screening unreliable.

### Improvements are strongest in the coldest-start regime

The largest transfer gains occurred at k = 3, where target-only models had the least target-building information. As k increased from 3 to 14 days, the absolute improvement decreased, suggesting that target-only models gradually benefit from additional target data. This pattern supports the interpretation that cross-building pretraining is most valuable when the target building is newly observed or has limited historical measurements.

### Forecasting gains should not be overinterpreted as anomaly-detection gains

Although transfer learning improved forecasting accuracy and residual stability, controlled injected-anomaly experiments showed that downstream detection performance depends strongly on the scoring and calibration strategy. In particular, improved residuals did not consistently translate into higher point-level AUPRC or F1. Therefore, anomaly detection is treated as an exploratory downstream application rather than the central claim of this paper.

## Cautious Claim Language

Use:

> Cross-building transfer learning consistently improves few-shot building energy forecasting and residual stability under cold-start target-data constraints.

Use:

> The benefits are strongest when only a few days of target-building data are available.

Use:

> Improved residual stability provides a better foundation for monitoring, but anomaly detection requires a calibrated detection layer and should be evaluated separately.

Avoid:

> Transfer learning solves building anomaly detection.

Avoid:

> Better forecasting guarantees better anomaly detection.

Avoid:

> The method universally outperforms all anomaly detection baselines.

