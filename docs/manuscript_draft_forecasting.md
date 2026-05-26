# Source Replay Fine-Tuning for Cross-Building Few-Shot Energy Load Forecasting Under Cold-Start Conditions

## Abstract

Accurate building energy load forecasting is important for operational monitoring, demand management, and energy-efficient control. However, newly deployed or sparsely monitored buildings often provide only a few days of target-specific data, making target-only model training unreliable. Cross-building transfer learning offers a practical way to reuse information from existing buildings, but its behavior under strict few-shot target adaptation remains insufficiently characterized, especially against strong non-neural baselines. This study evaluates cross-building transfer learning for hourly electricity forecasting using 12 buildings from the BDG2 dataset [1]. We compare target-only neural training, standard pretrain-then-fine-tune adaptation, and a source replay fine-tuning strategy for DLinear [6], termed DLinear-SRFT. Experiments cover three target-data budgets of 3, 7, and 14 days, two neural forecasting architectures, five random seeds, and multiple classical baselines, including persistence, Random Forest, Extra Trees, Ridge regression, and histogram gradient boosting. Main statistical tests are conducted at the building level after averaging repeated seeds within each target building. Standard cross-building pretraining consistently improves LSTM and DLinear over target-only training in forecasting accuracy and residual stability. DLinear-SRFT further reduces MAE relative to standard DLinear fine-tuning for all 12 target buildings at every evaluated data budget, with mean MAE reductions of 1.80, 1.39, and 1.59 at 3, 7, and 14 days, respectively. At 7 and 14 days, DLinear-SRFT becomes competitive with strong source+target tree baselines, while the 3-day ultra-cold-start setting remains challenging. These findings show that source replay is an effective adaptation mechanism for cold-start building forecasting and highlight the need to evaluate neural transfer methods against strong classical baselines.

## Keywords

Building energy forecasting; cold-start forecasting; transfer learning; few-shot learning; source replay; DLinear; BDG2

## 1. Introduction

Short-term building energy load forecasting supports a wide range of operational tasks, including energy management, demand response, fault screening, and model predictive control [2,3]. In many practical deployments, however, the buildings that most need predictive models are precisely those with limited historical data. New buildings, recently instrumented facilities, and buildings with interrupted metering records may provide only a few days of usable target-specific observations. In such cold-start settings, models trained only on the target building can be unstable, data hungry, and sensitive to initialization.

Cross-building transfer learning is a natural response to this problem. Existing buildings contain recurring temporal structures, occupancy-related patterns, weather-independent daily cycles, and load dynamics that may be partially reusable for a new target building. A forecasting model can therefore be pretrained on source buildings and adapted to a target building using a small amount of target data [4,5]. This idea is appealing, but it raises several empirical questions. First, it is not enough to show that transfer learning improves over a weak target-only neural baseline; practical forecasting studies must also compare against simple and classical methods that are often strong in short-horizon energy forecasting. Second, standard fine-tuning on only a few days of target data may over-adapt the pretrained model and forget useful source-domain structure. Third, repeated random seeds should not be treated as independent scientific samples when the real unit of generalization is the target building.

This paper studies cross-building transfer learning for few-shot hourly electricity forecasting under a strict cold-start protocol. We evaluate two neural forecasting architectures, LSTM and DLinear [6], using 12 target buildings from the BDG2 dataset [1] and three target-data budgets of 3, 7, and 14 days. The central comparison is between target-only few-shot training and cross-building pretraining followed by target adaptation. We then introduce a source replay fine-tuning strategy for DLinear, denoted DLinear-SRFT, which keeps a small source-domain loss during target adaptation:

`L = L_target + lambda_source * L_source`.

This design is intended to reduce over-adaptation to the tiny target set while preserving source-domain temporal structure learned during pretraining.

The contributions of this study are as follows:

1. We formulate a cold-start cross-building forecasting protocol for evaluating target-building adaptation when only 3, 7, or 14 days of target electricity data are available.
2. We compare target-only neural training, standard neural transfer, and source replay fine-tuning across 12 buildings, two neural architectures, and five random seeds.
3. We introduce DLinear-SRFT, a source replay adaptation strategy that consistently improves standard DLinear transfer across all target buildings in our experiments.
4. We benchmark neural transfer against strong simple and classical baselines, showing that DLinear-SRFT is competitive at 7 and 14 days but that the 3-day ultra-cold-start setting remains difficult.
5. We report building-level paired statistics after averaging repeated seeds within each target building, avoiding seed-level pseudoreplication in the main claims.

## 2. Related Work

### 2.1 Building Energy Load Forecasting

Building energy forecasting has been studied using statistical models, machine learning models, and deep neural networks. Reviews of data-driven building energy prediction show that model performance depends strongly on data availability, feature design, temporal resolution, and evaluation protocol [2,3]. Classical approaches such as persistence, autoregressive models, support vector regression, tree ensembles, and gradient boosting remain competitive in many short-horizon settings, particularly when load autocorrelation is strong and feature sets are limited. Deep learning models, including recurrent networks and temporal convolutional or transformer-based architectures, can capture nonlinear temporal dependencies but often require sufficient training data and careful regularization. In cold-start buildings, this data requirement becomes a central limitation.

### 2.2 Transfer Learning for Building Energy Modeling

Transfer learning aims to reuse information from source domains to improve performance in a target domain. For building energy modeling, transfer can occur across buildings, meters, climates, or operating regimes. Prior studies have reported that pretraining, domain adaptation, and multi-source transfer can reduce target-data requirements in data-scarce building prediction settings [4,5]. However, the benefit is not guaranteed because buildings may differ in scale, occupancy, equipment, and schedules. This makes rigorous source-target evaluation and negative-case reporting important.

### 2.3 Few-Shot and Cold-Start Forecasting

Few-shot forecasting differs from conventional forecasting because the target training set is intentionally small. In this regime, model capacity, initialization, and adaptation strategy can dominate performance. Simple baselines may be surprisingly strong because recent observations often provide high short-term predictive value. Therefore, claims about few-shot neural forecasting should be evaluated against persistence and classical source+target baselines, not only against target-only neural models. This requirement is especially important in building energy applications, where data scarcity and domain shift are common practical constraints [4,5].

### 2.4 Source Replay and Regularized Adaptation

Fine-tuning a pretrained model on a small target set can lead to overfitting and forgetting of source-domain structure. Source replay mitigates this issue by mixing source-domain examples or source-domain loss terms into target adaptation. This idea is related to continual learning and regularized transfer, where replay is used to preserve previously learned information while adapting to new data [7,8]. In this study, we test source replay as a simple adaptation mechanism for DLinear under cold-start building forecasting.

## 3. Methods

### 3.1 Problem Formulation

Let each building provide an hourly electricity time series with auxiliary calendar features. For a target building `b`, the forecasting task is to predict the next-hour electricity load from a fixed-length historical window:

`y_{t+1} = f(x_{t-W+1:t})`,

where `W = 24` hours in the main experiments. The input features include historical energy, hour of day, day of week, weekend indicator, and month. The output is the next-hour electricity value.

The cold-start setting assumes that only the first `k` days of the target building's training period are available for target adaptation, where `k` is 3, 7, or 14. All evaluation is performed on a later chronological test period. Source buildings are available for pretraining or source+target baselines, but the target test period is never used for scaling, training, thresholding, or model selection.

### 3.2 Data Splitting and Scaling

For each target building, the time series is split chronologically into training, validation, and test periods using a 60/20/20 ratio. The few-shot target set is selected from the beginning of the target training period. This design reflects deployment conditions in which a newly monitored building accumulates only a small amount of initial data before forecasting is required.

For target-only neural models, scalers are fit only on the available target few-shot windows. For transfer models, source pretraining uses source-building training data, while target fine-tuning uses scalers fit on the target few-shot windows. Classical source+target baselines are trained on source windows plus the target few-shot windows, with transformations fit only on training data. The target validation and test periods are transformed using training-fitted scalers.

### 3.3 Neural Forecasting Models

We evaluate two neural forecasting architectures. The first is an LSTM forecaster that maps a 24-hour input window to a one-step-ahead forecast using recurrent hidden states. The second is DLinear, a lightweight decomposition-linear architecture that separates trend and residual components using a moving average and applies linear projections to each input channel [6]. These two models provide complementary capacity profiles: LSTM represents a higher-capacity recurrent model, whereas DLinear provides a simpler linear temporal decomposition model.

### 3.4 Standard Cross-Building Transfer

The standard transfer method, denoted M3, consists of source pretraining followed by target fine-tuning. For a target building, all other selected buildings are treated as source buildings. A fresh model is pretrained on source-building training windows and validated on source validation windows. For `k > 0`, the pretrained model is then fine-tuned on the target building's `k`-day few-shot set and selected using target validation loss. The target-only baseline, denoted M2, trains the same architecture from scratch using only the target few-shot set.

### 3.5 DLinear-SRFT: Source Replay Fine-Tuning

Standard fine-tuning updates the pretrained model using only the small target set. This can over-adapt the model to a small number of target windows and reduce the benefit of source pretraining. To address this, we propose DLinear-SRFT, which adds source replay during target adaptation. During fine-tuning, each target batch is paired with a replayed source batch, and the optimization objective is:

`L = L_target + lambda_source * L_source`,

where `L_target` is the target few-shot forecasting loss, `L_source` is the source replay forecasting loss, and `lambda_source` controls the replay strength. We evaluate `lambda_source` values of 0.05, 0.1, and 0.2, and select 0.2 for the final DLinear-SRFT comparison based on aggregate performance. The replay mechanism is evaluated only as an adaptation strategy; the architecture remains the same as DLinear.

### 3.6 Classical and Simple Baselines

To avoid overstating neural transfer performance, we compare against several non-neural baselines:

- Persistence: the next-hour prediction equals the most recent observed load.
- Seasonal naive: the next-hour prediction equals the value from the same hour in the previous day.
- Ridge regression.
- Random Forest regression.
- Extra Trees regression.
- Histogram gradient boosting regression.

For supervised classical models, we evaluate both target-only and source+target pooled variants. Source+target variants train on source-building windows plus the target few-shot windows and therefore serve as strong practical baselines for cross-building forecasting.

## 4. Experimental Setup

### 4.1 Dataset

Experiments use hourly electricity data from 12 buildings selected from the BDG2 dataset [1]. The selected buildings include multiple building categories and are treated in a leave-one-building-out manner: each building is used as the target once, while the remaining buildings serve as sources. All models use the same chronological target split and the same test period for a given target building. Table 1 summarizes the selected buildings, building types, floor areas, time coverage, and electricity-load statistics.

### 4.2 Research Questions

The experiments answer three research questions:

**RQ1:** Does cross-building pretraining improve few-shot neural forecasting relative to target-only neural training?

**RQ2:** Does source replay fine-tuning further improve DLinear adaptation relative to standard pretrain-then-fine-tune adaptation?

**RQ3:** How competitive is neural transfer against simple and classical source+target baselines under different target-data budgets?

### 4.3 Metrics

Forecasting accuracy is evaluated using MAE, RMSE, and sMAPE. Residual stability is evaluated using `sigma_err`, the standard deviation of absolute prediction errors. Lower values are better for all four metrics. Residual stability is included because stable residuals are useful for operational monitoring and downstream screening, but anomaly detection is not treated as the central task in this paper.

### 4.4 Statistical Analysis

Repeated seeds are treated as repeated runs, not independent scientific samples. Therefore, the main statistical analysis first averages results across seeds within each target building and then performs paired tests across target buildings. For M3 versus M2 and DLinear-SRFT versus standard DLinear M3, we report mean paired differences, building-level win rates, bootstrap confidence intervals, and Wilcoxon signed-rank tests where appropriate. Seed-level summaries are reported only as supplementary evidence.

## 5. Results

### 5.1 Cross-Building Pretraining Improves Few-Shot Neural Forecasting

Across both neural architectures, source pretraining followed by target adaptation substantially improved cold-start forecasting relative to target-only training. After averaging repeated seeds within each target building, M3 achieved lower MAE than M2 for all 12 buildings across LSTM and DLinear at 3, 7, and 14 target-data days. Building-level Wilcoxon signed-rank tests confirmed significant MAE improvements in all six model-budget settings, with p-values around 0.0022. The largest absolute gains occurred at 3 target days, which is consistent with the expectation that source knowledge is most valuable when target-specific data are scarce.

Residual stability followed a similar pattern. M3 reduced `sigma_err` across nearly all model-budget settings, indicating that transfer learning did not only reduce average error but also made prediction residuals less variable. This finding supports the use of cross-building pretraining as a robust initialization strategy for cold-start neural forecasting.

### 5.2 Source Replay Consistently Improves DLinear Adaptation

DLinear-SRFT further improved the standard DLinear transfer model. With `lambda_source = 0.2`, DLinear-SRFT reduced MAE relative to standard DLinear M3 for every target building at every target-data budget. The mean MAE reductions were 1.803 at 3 days, 1.389 at 7 days, and 1.595 at 14 days, with 100% building-level win rates in all three settings.

The residual stability benefits were strongest at 7 and 14 days. At 7 days, DLinear-SRFT reduced `sigma_err` relative to standard DLinear M3 with a building-level win rate of 91.7%. At 14 days, the `sigma_err` win rate reached 100%. These results suggest that source replay helps DLinear retain useful source-domain structure during target adaptation, especially once the target set contains enough data to support meaningful fine-tuning.

### 5.3 Strong Classical Baselines Remain Competitive

The comparison with classical baselines shows that the cold-start forecasting problem is not solved simply by using neural transfer. Persistence and source+target tree models were highly competitive. In the 3-day setting, persistence and Random Forest source+target achieved the strongest overall performance, while neural methods remained less competitive. This result indicates that ultra-cold-start forecasting can be dominated by short-term autocorrelation and that complex adaptation may not be reliable with only three days of target data.

At 7 and 14 days, DLinear-SRFT entered the top-performing group. At 7 days, DLinear-SRFT achieved the best mean rank among the selected methods, while Random Forest source+target had a slightly lower mean MAE. At 14 days, DLinear-SRFT again achieved the best mean-rank profile and a mean MAE close to Random Forest source+target. These results support a conditional claim: source replay neural transfer is most useful once the target building provides at least one week of data, while simpler or pooled classical methods remain strong competitors in the most data-scarce setting.

### 5.4 Architecture Dependence of Source Replay

Source replay did not improve all neural architectures. While DLinear-SRFT consistently improved DLinear, the same replay strategy generally worsened LSTM MAE relative to standard LSTM pretrain-then-fine-tune adaptation. This suggests that replay fine-tuning is architecture-dependent. A plausible explanation is that DLinear's lower-capacity decomposition-linear structure benefits from source regularization, whereas the higher-capacity LSTM may require a different replay schedule, layer-freezing strategy, or learning-rate design. For this reason, the final proposed method is DLinear-SRFT rather than a generic replay strategy for all neural models.

## 6. Discussion

The results provide evidence that cross-building transfer learning is a useful strategy for cold-start building energy forecasting, but they also clarify its limits. Standard pretraining consistently improves neural models over target-only training, demonstrating that source-building information can provide a strong initialization when target data are scarce. However, strong classical baselines show that neural transfer should not be evaluated only against weak target-only models. In short-horizon building electricity forecasting, persistence and tree ensembles can be difficult to beat, especially when only a few target days are available.

DLinear-SRFT improves the transfer pipeline by adding source replay during target adaptation. The consistent building-level improvement over standard DLinear fine-tuning suggests that replay reduces over-adaptation to the tiny target set. This is particularly useful at 7 and 14 target days, where the target data contain enough information for adaptation but still benefit from source-domain regularization. The weaker performance at 3 days indicates that the most extreme cold-start condition may require simpler assumptions, stronger priors, or alternative adaptation mechanisms.

The architecture-dependent behavior of replay is also important. LSTM did not benefit from the same replay strategy, which means source replay should not be presented as a universal neural adaptation mechanism. Instead, the evidence supports a more specific conclusion: source replay is effective for DLinear in this cold-start forecasting protocol. This narrower claim is more defensible and better aligned with the empirical results.

Although improved residual stability may be useful for downstream monitoring, this paper does not claim to solve anomaly detection. Controlled anomaly-screening experiments showed that better forecasting residuals do not automatically produce better detection metrics. Therefore, anomaly detection should be treated as a downstream task requiring separate scoring, calibration, and validation.

## 7. Limitations

This study has several limitations. First, the evaluation uses 12 target buildings rather than the full BDG2 building set. The selected buildings provide a meaningful cross-building protocol, but broader building coverage would further strengthen external validity. Second, the study focuses on hourly electricity forecasting with calendar features and does not evaluate all possible weather, occupancy, or operational covariates. Third, DLinear-SRFT improves DLinear but not LSTM, so the replay mechanism is not architecture-universal. Fourth, in the most extreme 3-day setting, persistence and source+target tree baselines remain stronger than the proposed neural transfer method on average. Finally, anomaly detection is outside the central contribution and requires separate evaluation with appropriate labels and detection metrics.

## 8. Conclusion

This study evaluated cross-building transfer learning for few-shot building electricity forecasting under cold-start conditions. Standard source pretraining followed by target fine-tuning consistently improved LSTM and DLinear over target-only training across 12 target buildings and three target-data budgets. We further introduced DLinear-SRFT, a source replay fine-tuning strategy that retains a weighted source loss during target adaptation. DLinear-SRFT improved over standard DLinear transfer for every target building at 3, 7, and 14 target days, and became competitive with strong source+target tree baselines at 7 and 14 days. The results support source replay as a useful adaptation mechanism for DLinear-based cold-start forecasting, while also showing that ultra-cold-start forecasting and architecture-general replay remain open challenges.

## Figure Captions

**Figure 1. Forecasting performance of selected methods across target-data budgets.** MAE is reported for persistence, source+target tree baselines, target-only neural models, standard neural transfer, and DLinear-SRFT at 3, 7, and 14 target days. Error bars indicate standard errors across repeated building-seed runs.

**Figure 2. DLinear transfer curve under increasing target-data budgets.** The figure compares target-only DLinear, standard DLinear pretrain-then-fine-tune, DLinear-SRFT, persistence, and Random Forest source+target. DLinear-SRFT closes much of the gap to strong source+target classical baselines at 7 and 14 days.

**Figure 3. Building-level effect of source replay.** Bars show DLinear-SRFT minus standard DLinear M3 MAE for each target building. Negative values indicate that source replay improves forecasting. DLinear-SRFT improves MAE for all 12 target buildings across the evaluated target-data budgets.

## Citation Placeholders To Fill

- See `docs/references_forecasting.md` for the current numbered reference list.
