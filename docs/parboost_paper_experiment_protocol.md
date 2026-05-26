# Paper Experiment Protocol: Cold-Start Building Load Forecasting

## Working Title

Rethinking Transfer Learning for Cold-Start Building Load Forecasting: Strong Baselines, Failure Modes, and Persistence-Anchored Residual Boosting

## Central Research Question

In cold-start building electricity load forecasting, when does transfer learning help, why do direct neural transfer methods fail, and can persistence-anchored residual adaptation provide a more robust alternative?

This paper should not be framed as an anomaly-detection paper or as a universal neural-transfer SOTA paper. The defensible framing is a cold-start forecasting benchmark, failure-mode analysis, and mitigation method.

## Main Claims

1. Persistence and source-target tree models are strong cold-start baselines and must be included in any credible evaluation.
2. Direct neural transfer can suffer severe negative transfer, especially under near-zero/inactive target loads and source-target mismatch.
3. Persistence-anchored residual adaptation reduces failure risk by learning only the residual not explained by persistence.
4. PARBoost ranks first on the BDG2 expanded benchmark by mean rank and remains competitive on COFACTOR, but it does not universally dominate ExtraTrees-ST/RF-ST on external active buildings.

## Method To Present

Use the name `PARBoost`.

Full name:

Persistence-Anchored Residual Boosting

Prediction form:

```text
y_hat[t+1] = y[t] + alpha_target * s_target * f_theta(x[t])
```

where:

- `y[t]` is the persistence anchor.
- `f_theta` predicts a normalized residual.
- `s_target` rescales the normalized residual to the target building.
- `alpha_target` is selected from target validation data only.
- Source samples are weighted by similarity computed from target few-shot data only.

Do not describe `alpha_target` as an ensemble weight between models. It is a target-side residual calibration factor that controls correction magnitude.

## Datasets

### BDG2 Expanded Benchmark

Main benchmark:

- 24 buildings.
- Chronological split: train 60%, validation 20%, test 20%.
- Few-shot target data: first `k` days of target training split.
- k values: 3, 7, 14, 30.
- Seeds: 42, 43, 44, 45, 46.

Purpose:

- Main ranking.
- Failure-mode analysis.
- Active/inactive diagnosis.
- Negative transfer analysis.

Current result location:

- `results/parboost_expanded_bdg2/table_par_combined_ranking.csv`
- `results/parboost_expanded_bdg2/table_par_pairwise_deltas.csv`

### COFACTOR External Active-Building Pilot

External validation:

- 12 active buildings currently evaluated.
- k values: 3, 7, 14, 30.
- Seeds: 42, 43, 44, 45, 46.

Purpose:

- Test whether the conclusions generalize beyond BDG2.
- Show that PARBoost is robust relative to Persistence and neural transfer.
- Acknowledge that ExtraTrees-ST/RF-ST remain stronger on this active dataset.

Current result location:

- `results/parboost_cofactor_pilot/table_par_combined_ranking.csv`
- `results/parboost_cofactor_pilot/table_par_pairwise_deltas.csv`

## Experimental Questions

### RQ1: How strong are simple and tree-based baselines in cold-start forecasting?

Methods:

- Persistence.
- Seasonal naive.
- Ridge target-only.
- Ridge source-target.
- RF source-target.
- ExtraTrees source-target.
- HistGBDT source-target.
- DLinear M2 target-only.
- DLinear M3 pretrain+fine-tune.
- DLinear SRFT.

Required outputs:

- Leaderboard by k.
- Mean MAE.
- Median MAE.
- Worst-case MAE.
- Mean rank.
- Top-3 rate.

Expected interpretation:

Persistence and source-target tree models are strong enough that neural transfer cannot be evaluated only against target-only neural baselines.

### RQ2: When and why does direct neural transfer fail?

Analyses:

- Active vs inactive / near-zero split.
- Target zero ratio.
- Target test variance.
- Persistence MAE quantiles.
- Source-target similarity.
- Building type.
- Worst-case error.
- Negative transfer rate.

Required outputs:

- Near-zero building audit table.
- Failure cases ranked by neural-transfer error.
- Correlation table between target statistics and transfer failure.
- Negative-transfer table by method and k.

Expected interpretation:

Neural transfer fails when residual dynamics are weak, target load is near constant, or source-target mismatch causes the model to learn corrections that persistence did not need.

### RQ3: Does PARBoost reduce failure risk?

Primary comparison:

- PARBoost vs Persistence.
- PARBoost vs RF-ST.
- PARBoost vs ExtraTrees-ST.
- PARBoost vs HistGBDT-ST.
- PARBoost vs DLinear-SRFT.
- PARBoost vs PA-SRFT.

Primary metrics:

- Mean rank.
- Top-3 rate.
- Worst-case MAE.
- Negative transfer rate.
- Paired building-level delta.

BDG2 result to emphasize:

`PAR_HIST_GBDT_SW_CAL` ranks first by mean rank across k=3/7/14/30 on the 24-building benchmark.

COFACTOR result to state cautiously:

PARBoost improves over Persistence and neural transfer, but ExtraTrees-ST/RF-ST remain stronger on this active external dataset.

### RQ4: Which components of PARBoost matter?

Required ablation:

1. HistGBDT-ST direct load prediction.
2. Residual target without similarity weighting.
3. Residual target plus similarity weighting.
4. Residual target plus target-side calibration.
5. Full PARBoost.

Metrics:

- Mean MAE.
- Mean rank.
- Worst-case MAE.
- Negative transfer rate vs Persistence.
- Negative transfer rate vs RF-ST.

Purpose:

Prevent the reviewer objection that PARBoost is only a tuned tree model.

## Main Tables

### Table 1: Dataset and Benchmark Construction

Columns:

- Dataset.
- Raw buildings.
- Eligible buildings.
- Benchmark buildings.
- Train/val/test split.
- k values.
- Seeds.
- Notes on active/inactive buildings.

This table must explain why not all raw buildings are used indiscriminately.

### Table 2: BDG2 Main Leaderboard

Rows:

- Persistence.
- RF-ST.
- ExtraTrees-ST.
- HistGBDT-ST.
- DLinear-SRFT.
- PA-SRFT.
- PARBoost.

Columns:

- k.
- Mean MAE.
- Median MAE.
- Worst-case MAE.
- Mean rank.
- Top-3 rate.

### Table 3: COFACTOR External Validation

Same structure as Table 2.

The discussion must acknowledge that PARBoost is not first on COFACTOR.

### Table 4: Negative Transfer and Worst-Case Error

Rows:

- DLinear-M3.
- DLinear-SRFT.
- PA-SRFT.
- RF-ST.
- ExtraTrees-ST.
- PARBoost.

Columns:

- Negative transfer rate vs Persistence.
- Negative transfer rate vs RF-ST.
- Worst-case MAE.
- Paired median delta.

### Table 5: PARBoost Ablation

Rows:

- Direct HistGBDT-ST.
- Residual only.
- Residual + similarity.
- Residual + calibration.
- Full PARBoost.

Columns:

- Mean rank.
- Mean MAE.
- Worst-case MAE.
- Negative transfer rate.

## Main Figures

### Figure 1: Experimental Design

Diagram:

Raw hourly load -> chronological split -> few-shot target adaptation -> source-target transfer -> test evaluation.

Show that target test data is never used for similarity or calibration.

### Figure 2: BDG2 Leaderboard by k

Line or grouped bar plot:

- x-axis: k.
- y-axis: mean rank or mean MAE.
- Methods: Persistence, RF-ST, ExtraTrees-ST, DLinear-SRFT, PARBoost.

### Figure 3: Failure-Mode Scatter

Plot:

- x-axis: Persistence MAE or target load variance.
- y-axis: DLinear-SRFT minus Persistence MAE.
- Color: active/inactive or building type.

Purpose:

Show where direct neural transfer collapses.

### Figure 4: Paired Delta Distribution

Violin/box plot:

- PARBoost minus baseline MAE.
- Baselines: Persistence, RF-ST, ExtraTrees-ST, DLinear-SRFT.

Separate panels for BDG2 and COFACTOR.

### Figure 5: Ablation Result

Grouped bars:

- Mean rank.
- Worst-case MAE.
- Negative transfer rate.

## Reviewer Risks And Defenses

### Benchmark Reproduction Strategy

It is acceptable to discuss existing building-load forecasting benchmarks and, where feasible, reproduce representative methods from prior papers under the present cold-start protocol. However, reported numbers from prior papers should not be copied directly into the main leaderboard unless the data cleaning, building subset, chronological split, forecast horizon, feature set, scaling policy, and evaluation metrics are exactly identical.

Main-paper comparisons should therefore use a unified reimplementation protocol:

- All baselines and proposed methods use the same train/validation/test split.
- All methods use the same target few-shot budget.
- All source-target methods have the same source-building access.
- Similarity, calibration, and model selection use only training or validation data.
- The target test period is used only for final evaluation.

Prior-paper results can be used in Related Work or Supplementary Material to contextualize the field, but not as direct evidence of superiority unless reproduced under the unified protocol. A useful supplementary table should compare prior studies by dataset, horizon, cold-start setting, baselines, metrics, and whether persistence/RF/ExtraTrees baselines were included.

### Risk 1: Why not use the entire dataset?

Defense:

Cold-start forecasting requires valid chronological train/validation/test windows and non-degenerate target loads. The main benchmark uses a pre-specified eligible and stratified subset for fair comparison with computationally expensive neural baselines. Inactive/near-zero buildings are not hidden; they are analyzed explicitly as a failure mode. A full-eligible classical/PARBoost robustness check should be added if time permits.

Recommended additional experiment:

Run Persistence, RF-ST, ExtraTrees-ST, HistGBDT-ST, and PARBoost on all eligible active BDG2 buildings.

### Risk 2: PARBoost is just a tuned tree model.

Defense:

Add ablation showing that residual anchoring and target-side calibration are responsible for reduced failure risk. Compare directly against direct HistGBDT-ST and ExtraTrees-ST.

### Risk 3: PARBoost is not first on COFACTOR.

Defense:

Do not claim universal superiority. State that COFACTOR validates robustness relative to Persistence and neural transfer, while confirming that source-target tree baselines are exceptionally strong on active-building datasets.

### Risk 4: Seeds are treated as independent samples.

Defense:

Aggregate seeds within each building-k-method unit. Use paired building-level comparisons as the statistical unit.

### Risk 5: Similarity or calibration leaks test information.

Defense:

Similarity must use source training data and target few-shot data only. Calibration must use target validation data only. Test data is used only once for final evaluation.

## Statistical Testing

Use paired tests over building-level results, not seed-level rows.

Recommended:

- Wilcoxon signed-rank test for paired MAE deltas.
- Bootstrap confidence intervals over buildings.
- Report effect sizes as median paired delta and win rate.

Do not report p-values over seeds as if seeds were independent buildings.

## Safe Abstract Claim

Cold-start building load forecasting is often evaluated through transfer learning, yet the strength of simple persistence and tree-based source-target baselines is underexplored. We construct a strict cold-start benchmark on BDG2 and evaluate an external active-building dataset from COFACTOR. The results show that direct neural transfer can suffer severe negative transfer and worst-case failures, especially when persistence is already strong or source-target mismatch is high. To mitigate these failures, we propose PARBoost, a persistence-anchored residual boosting approach with similarity-weighted source transfer and target-side residual calibration. PARBoost ranks first by mean rank on the BDG2 benchmark and consistently improves over persistence and neural transfer on COFACTOR, although source-target tree ensembles remain the strongest external active-building baselines. These findings suggest that robust cold-start forecasting requires explicit persistence anchoring and strong classical baselines rather than direct neural fine-tuning alone.

## Immediate Next Steps

1. Generate Table 2 and Table 3 from the existing ranking files.
2. Generate negative-transfer tables from paired building-level deltas.
3. Implement the PARBoost ablation.
4. Create active/inactive and near-zero diagnostic tables.
5. Draft the Introduction and Experiment sections using this protocol.
