# Manuscript Blueprint: Cold-Start Building Energy Forecasting

## Working Title

Source replay fine-tuning for cross-building few-shot energy load forecasting under cold-start conditions

## Short Title

Source replay transfer for cold-start building load forecasting

## Central Claim

Cross-building transfer learning substantially improves few-shot neural building energy forecasting compared with target-only training, and a source replay fine-tuning strategy further strengthens DLinear adaptation, making it competitive with strong source+target tree baselines at moderate few-shot budgets.

## Method Name

**DLinear-SRFT**: DLinear with Source Replay Fine-Tuning.

During target adaptation, DLinear-SRFT fine-tunes the source-pretrained DLinear model on target few-shot windows while replaying source-building batches:

`L = L_target + lambda_source * L_source`

The final selected value is:

`lambda_source = 0.2`

## Abstract Draft

Accurate building energy load forecasting is difficult for newly deployed or sparsely monitored buildings because only a few days of target-building data may be available. Cross-building transfer learning offers a practical way to reuse information from existing buildings, but its effectiveness under strict few-shot target adaptation remains underexplored, especially against strong classical source+target baselines. This study evaluates cross-building transfer learning for hourly electricity forecasting using 12 buildings from the BDG2 dataset. We compare target-only training, standard pretrain-then-fine-tune adaptation, and a proposed source replay fine-tuning strategy for DLinear, termed DLinear-SRFT. Experiments cover three target-data budgets of 3, 7, and 14 days, two neural architectures, five random seeds, and multiple classical baselines including persistence, Random Forest, Extra Trees, Ridge regression, and histogram gradient boosting. Main statistical tests are performed at the building level after averaging repeated seeds within each target building. Standard pretraining consistently improves LSTM and DLinear over target-only training in MAE and residual error variability. DLinear-SRFT further reduces MAE relative to standard DLinear fine-tuning for all 12 target buildings at every evaluated data budget, with mean MAE reductions of 1.80, 1.39, and 1.59 at k = 3, 7, and 14 days, respectively. At k = 7 and k = 14, DLinear-SRFT becomes competitive with strong source+target tree baselines, while k = 3 remains challenging and is often better served by simple persistence or pooled tree models. These findings show that source replay is an effective adaptation mechanism for cold-start building forecasting, while also highlighting the importance of strong non-neural baselines in few-shot energy prediction studies.

## Contributions

1. We formulate a cold-start cross-building forecasting protocol that evaluates target-building adaptation when only 3, 7, or 14 days of target electricity data are available.
2. We provide a building-level empirical comparison of target-only training, standard neural transfer, and source replay fine-tuning across 12 buildings, two neural forecasting architectures, and five random seeds.
3. We introduce DLinear-SRFT, a source replay fine-tuning variant that reduces over-adaptation to the tiny target set by retaining a weighted source loss during fine-tuning.
4. We benchmark neural transfer methods against strong classical baselines, showing that DLinear-SRFT is competitive at k = 7 and k = 14 but that ultra-cold-start k = 3 remains difficult.
5. We report residual stability and building-level paired statistics, avoiding seed-level pseudoreplication in the main evidence.

## Main Results To Emphasize

### Result 1: Standard transfer beats target-only neural training

Use the original M3 vs M2 building-level statistics:

- Both LSTM and DLinear improve over target-only training across k = 3, 7, and 14.
- Building-level MAE win rate is 100% in all six model/k settings.
- Building-level Wilcoxon p-value is approximately 0.0022 for MAE across all six settings.
- Residual error variability (`sigma_err`) also improves consistently.

Interpretation:

> Cross-building pretraining provides a strong initialization for few-shot target adaptation.

### Result 2: DLinear-SRFT consistently improves over standard DLinear transfer

Use `table_final_dlinear_m3r_vs_m3.md`:

- k = 3: MAE delta = -1.803, building win rate = 100%.
- k = 7: MAE delta = -1.389, building win rate = 100%.
- k = 14: MAE delta = -1.595, building win rate = 100%.
- sigma_err improves most clearly at k = 7 and k = 14.

Interpretation:

> Source replay reduces the risk of overfitting the tiny target set during fine-tuning and is particularly effective for DLinear.

### Result 3: Strong classical baselines are competitive

Use `table_final_leaderboard.md`:

- k = 3:
  - Persistence and RF source+target are strongest.
  - DLinear-SRFT improves over DLinear M3 but does not dominate.
- k = 7:
  - DLinear-SRFT has the best mean rank among selected methods.
  - RF source+target has slightly lower mean MAE.
- k = 14:
  - DLinear-SRFT has the best mean rank and is close to RF source+target in mean MAE.

Interpretation:

> The proposed neural transfer method is not universally superior, but it becomes highly competitive once the target building provides at least one week of data.

## Recommended Main Figures

### Figure 1

`figure_final_method_mae.png`

Purpose:

Show all selected methods across k = 3, 7, and 14.

Main message:

Strong classical baselines matter; neural transfer is not compared only against weak baselines.

### Figure 2

`figure_final_dlinear_transfer_curve.png`

Purpose:

Show target-only DLinear, standard DLinear transfer, DLinear-SRFT, RF source+target, and persistence across k.

Main message:

DLinear-SRFT closes much of the gap to strong source+target baselines at k = 7 and k = 14.

### Figure 3

`figure_final_m3r_delta_k3.png`, `figure_final_m3r_delta_k7.png`, `figure_final_m3r_delta_k14.png`

Purpose:

Show building-level DLinear-SRFT minus DLinear-M3 MAE deltas.

Main message:

DLinear-SRFT improves over standard DLinear transfer consistently across buildings.

## Results Section Draft

### Cross-building pretraining improves few-shot neural forecasting

Across both neural architectures, source pretraining followed by target adaptation substantially improved cold-start forecasting relative to target-only training. After averaging repeated seeds within each target building, M3 achieved lower MAE than M2 for all 12 buildings across all evaluated model and data-budget settings. Building-level Wilcoxon signed-rank tests confirmed that these improvements were statistically significant for both LSTM and DLinear at k = 3, 7, and 14 days. The improvements were largest at k = 3, consistent with the expectation that source knowledge is most valuable when target-specific data are scarce.

### Source replay improves DLinear adaptation

Standard fine-tuning updates the pretrained model using only a small target set, which can cause over-adaptation and partial forgetting of source-domain structure. To address this, we evaluated DLinear-SRFT, which retains a weighted source replay loss during target adaptation. With lambda_source = 0.2, DLinear-SRFT reduced MAE relative to standard DLinear pretrain-then-fine-tune for every target building at k = 3, k = 7, and k = 14. The mean MAE reductions were 1.803, 1.389, and 1.595, respectively. Residual variability was also reduced, especially at k = 7 and k = 14. These results indicate that source replay provides a stable adaptation mechanism for DLinear in few-shot building forecasting.

### Comparison with strong classical baselines

The advantage of neural transfer depends on the comparison baseline. Source+target Random Forest and Extra Trees baselines were strong competitors, and simple persistence was difficult to beat in the most extreme k = 3 setting. At k = 7 and k = 14, however, DLinear-SRFT entered the top-performing group. At k = 7, DLinear-SRFT achieved the best mean rank among selected methods, while Random Forest source+target achieved the lowest mean MAE. At k = 14, DLinear-SRFT again achieved the best mean rank and a mean MAE close to Random Forest source+target. These findings suggest that source replay neural transfer is most useful once the target building provides at least one week of data, whereas the ultra-cold-start case remains challenging.

## Discussion Points

- The paper's contribution is not a new forecasting architecture; it is a cold-start transfer protocol and a replay-based adaptation mechanism.
- DLinear benefits more from source replay than LSTM, suggesting that replay regularization is better matched to the lower-capacity linear decomposition model.
- Persistence is surprisingly strong at k = 3, which is plausible for hourly electricity with short-term autocorrelation.
- Source+target tree baselines should be treated as serious competitors, strengthening the credibility of the evaluation.
- Anomaly detection should remain outside the central claim.

## Limitations

- The study uses 12 target buildings rather than the full BDG2 building set.
- Real-world deployment may involve missing data, meter changes, and operational regime shifts not fully represented here.
- DLinear-SRFT improves DLinear but does not improve LSTM, so the replay mechanism is architecture-dependent.
- At k = 3, simple and classical methods remain stronger than neural transfer on average.
- The work evaluates forecasting; anomaly detection requires separate detection-layer design and real or carefully controlled anomaly labels.

## Safe Claim Language

Use:

> DLinear-SRFT consistently improves over standard DLinear transfer across all target buildings.

Use:

> The proposed replay strategy is competitive with strong source+target tree baselines at k = 7 and k = 14.

Use:

> Ultra-cold-start forecasting with only three days of target data remains challenging.

Avoid:

> The proposed method universally outperforms all baselines.

Avoid:

> Neural transfer is always better than classical forecasting methods.

Avoid:

> Forecasting improvements imply better anomaly detection.

