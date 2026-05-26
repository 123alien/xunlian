# Persistence-Anchored Multi-Scale Residual Transfer for Cold-Start Building Load Forecasting

## Title Page

Author names, affiliations, corresponding author information, and ORCID identifiers should be completed before submission.

Corresponding author: [Name, email, institution]

## Highlights

- Strong persistence and tree baselines reshape cold-start transfer evaluation.
- Direct neural transfer can produce severe worst-case building-level failures.
- PA-MSR learns multi-scale residual corrections around a persistence anchor.
- PA-MSR+ is useful mainly when the target adaptation window is longer.
- External COFACTOR validation supports severity-aware residual transfer.

## Graphical Abstract

Suggested graphical abstract file: `figures/pa_msr/fig1_protocol_no_leakage.svg`.

## Abstract

Cold-start building electricity forecasting is required when newly monitored or sparsely instrumented buildings provide only a few days of target-specific data. Cross-building transfer learning is a natural response to this setting, but its value can be overstated when evaluation is dominated by weak target-only neural baselines. This study re-examines next-hour electricity forecasting under a strict chronological protocol in which persistence and source-target tree ensembles are treated as first-class baselines. The experiments show that these baselines are highly competitive, while direct neural transfer can produce severe worst-case failures. We propose persistence-anchored multi-scale residual transfer (PA-MSR), which learns the residual not explained by persistence rather than the absolute next-hour load. PA-MSR combines persistence anchoring, residual RevIN, and multi-scale residual encoding; PA-MSR+ adds daily-profile source selection and target-side residual calibration for longer adaptation windows. On the 24-building BDG2 benchmark, PA-MSR or PA-MSR+ obtains the lowest mean MAE for all k in {3, 7, 14, 30}. On a 120-building active BDG2 robustness subset, PA-MSR remains the top-ranked method by mean MAE and mean rank across all k values. On the external COFACTOR-44 dataset, PA-MSR is strongest at k=3/7, while PA-MSR+ is strongest at k=14/30. Module analysis identifies residual RevIN and multi-scale residual encoding as stable contributors, whereas ordinary similarity weighting and residual gates are not reliable. The supported claim is therefore specific: persistence anchoring, residual normalization, and multi-scale residual structure provide a robust mechanism for cold-start building load forecasting, but no single variant universally dominates.

## Keywords

Building load forecasting; cold-start forecasting; transfer learning; residual learning; persistence baseline; multi-scale residuals; BDG2; COFACTOR

## 1. Introduction

Short-term building electricity forecasting supports demand response, operational monitoring, model predictive control, and data-driven energy management [1,2,15,16]. In mature deployments, a forecasting model can be trained from months or years of target-building data. Many practical deployments are less favorable. A building may be newly commissioned, recently instrumented, or affected by missing and interrupted metering records. In these cases, the forecasting problem is cold-start rather than data-rich: only a few days of target-specific observations are available, but reliable next-hour forecasts are still required.

Cross-building transfer learning is a natural response to this limitation. Source buildings may contain load patterns, calendar responses, and operating regimes that have not yet appeared in the target building. This has motivated source pretraining, fine-tuning, pooled source-target training, and replay-based neural transfer strategies for building energy prediction [3-5,17,18]. However, cold-start building forecasting also has strong simple baselines that are easy to underestimate. For next-hour electricity forecasting, persistence can be highly competitive because many building loads are strongly autocorrelated. Classical tree ensembles trained on source and few-shot target windows can also be effective in small-data regimes. A transfer method that improves over a target-only neural model may therefore still fail against the predictors that an operational benchmark should include.

This baseline issue changes the scientific question. The relevant question is not whether transfer learning can outperform a weak cold-start neural model. It is whether transfer remains useful when persistence and source-target tree ensembles are evaluated under the same information budget. This also raises a mechanism question. A direct transfer model may fail not because the model class is inherently weak, but because it is asked to transfer absolute load levels across heterogeneous buildings. Buildings can differ substantially in scale, activity level, zero-load behavior, daily profile, and residual dynamics. Directly transferring absolute next-hour load values can therefore produce negative transfer, especially when persistence already explains much of the test-period variation.

We address this problem by reframing cold-start transfer around the residual left by persistence. The proposed framework predicts the next-hour load as the current load plus a learned residual correction. Its default variant, PA-MSR, combines persistence anchoring, residual RevIN for instance-level distribution shift, and multi-scale residual features that summarize short-, medium-, and daily-scale changes. A second variant, PA-MSR+, adds daily-profile source selection and target-side residual calibration. This enhanced variant is treated as conditional rather than universally superior, because it requires enough target data to estimate profile and calibration statistics. The design changes the supervised target itself rather than combining two absolute-load predictors after training.

We evaluate this idea under a strict chronological cold-start protocol using two complementary datasets. A 24-building BDG2 benchmark provides the full method matrix, including persistence, source-target tree ensembles, neural transfer baselines, prior residual boosting variants, PA-MSR, and PA-MSR+. A 120-building active BDG2 robustness subset tests whether PA-MSR remains stable under a larger active-building sample, while retaining the boundary that the full neural matrix is not repeated at this scale. An external 44-building COFACTOR validation set tests whether the multi-scale residual variants generalize outside BDG2. This design is deliberately baseline-aware: each empirical claim is made only against methods evaluated under the same splits and target-data budgets.

The contributions of this study are:

1. We construct a strict cold-start evaluation protocol showing that persistence and source-target tree ensembles are strong baselines for next-hour building electricity forecasting.
2. We show that direct neural transfer can suffer large worst-case errors and negative transfer, motivating residual rather than absolute-load transfer.
3. We introduce PA-MSR, a persistence-anchored residual transfer method that combines residual RevIN with multi-scale residual encoding.
4. We evaluate an enhanced PA-MSR+ variant with daily-profile source selection and residual calibration, showing that it is useful mainly in moderate cold-start settings rather than under extreme k=3 adaptation.
5. We provide module, paired, large-sample, and stratified analyses showing that residual anchoring, RevIN, and MSR are the supported core mechanisms, whereas ordinary similarity weighting and residual gates should not be treated as universal improvements.

## 2. Related Work

### 2.1 Building Load Forecasting

Building load forecasting has been studied across statistical, machine-learning, and deep-learning paradigms. Traditional forecasting models and classical machine-learning methods remain widely used because they are transparent, data-efficient, and easy to combine with calendar variables, lagged loads, and weather or occupancy covariates when those features are available [1,2]. Tree ensembles are particularly attractive in applied forecasting because they handle nonlinear interactions and mixed feature types without requiring large target-specific datasets [8-10].

Deep forecasting models, including recurrent, convolutional, and transformer-style architectures, can represent complex temporal dependencies and multi-scale seasonal structure [7,19-22]. Their advantages, however, depend on the amount of training data, the feature set, the target-building split, and the strength of the baselines. In cold-start building forecasting, the target building may contribute only a few days of observations. This small target-data budget changes the modeling problem. High-capacity models may overfit the target building, while simple predictors can remain competitive because next-hour electricity loads often have strong short-term autocorrelation.

### 2.2 Transfer Learning for Building Energy Prediction

Transfer learning attempts to reuse information from source buildings to improve prediction in a data-poor target building. Common strategies include source pretraining followed by target fine-tuning, source-target pooled training, domain adaptation, representation learning, and source replay during target adaptation [3-5,17,18]. These approaches are appealing because source buildings may contain calendar responses, operational rhythms, and load-shape patterns that are not yet visible in the target building.

The central risk is negative transfer. Source and target buildings may differ in scale, meter type, occupancy schedule, equipment operation, zero-load behavior, and residual dynamics. Under this mismatch, a transferred model may learn corrections that are useful for source buildings but harmful for the target building. Negative transfer is particularly important in cold-start settings because the few-shot target data may be insufficient to override misleading source information. A rigorous evaluation therefore needs to report not only mean accuracy, but also paired building-level failures, worst-case errors, and negative-transfer rates relative to operational baselines.

### 2.3 Persistence, Residual Learning, and Forecast Anchoring

Persistence is often treated as a simple baseline, but in next-hour building electricity forecasting it can be a strong operational reference. When loads are highly autocorrelated or nearly constant over the test period, predicting the most recent observed load can be difficult to beat. A model that directly predicts absolute load must therefore learn both the dominant persistence component and the smaller residual component. In heterogeneous cross-building transfer, this can be inefficient and unstable because absolute load levels and scales differ substantially across buildings.

Residual learning offers a different formulation. Rather than predicting the full target value, the model predicts the part not already explained by a simple reference forecast. This idea is related to boosting and error-correction models, where a learner focuses on remaining error after a baseline predictor [10]. In the present cold-start setting, the reference forecast is persistence, and the residual target is the next-hour change in load. This formulation aligns the model with the practical question: whether source information can improve the residual beyond what persistence already explains.

Persistence-anchored residual transfer differs from a post-hoc ensemble between persistence and a second absolute-load predictor. In an ensemble, both components typically forecast the same target variable and are combined after training. In PA-MSR, persistence defines the supervised target during training, and the model is trained only on normalized residual corrections. When calibration is used in PA-MSR+, it controls the amplitude of this residual correction rather than weighting two independently trained absolute-load models.

### 2.4 Benchmarking and Strong Baselines

Benchmarking is especially important in applied forecasting because reported improvements can depend on data cleaning, building selection, split policy, forecast horizon, feature availability, scaling, and metric choice [11,18]. In cold-start transfer, additional protocol details matter. The target building must be excluded from the source set, few-shot target data must come only from the permitted adaptation period, normalization and source selection must not use test information, and seeds should not be treated as independent buildings.

Strong baselines are also essential. A method that improves over target-only neural training may still be practically weak if it fails against persistence, Random Forest, ExtraTrees, or gradient boosting trained under the same source-target data budget. Existing studies can provide useful methods and evaluation ideas, but reported numbers should not be copied into a new leaderboard unless the data cleaning, building subset, chronological split, forecast horizon, feature set, scaling policy, and metrics are identical. In this study, direct claims rely only on methods evaluated under a unified protocol. Prior work is used to motivate the task and baseline set, while empirical comparisons are based on re-evaluated methods with the same splits, target-data budgets, and statistical unit.

### 2.5 Position of This Study

This study contributes at the intersection of cold-start building forecasting, transfer-learning evaluation, and residual error correction. Its novelty is not the claim that tree boosting is universally superior to neural forecasting, nor that persistence can be ignored. Instead, the paper makes three narrower claims. First, cold-start transfer should be evaluated against strong persistence and source-target tree baselines. Second, direct transfer of absolute load dynamics can produce severe worst-case failures under building heterogeneity. Third, persistence-anchored residual transfer provides a practical way to reduce this failure mode. The COFACTOR results further show that the final multi-scale residual variants can generalize beyond BDG2, while remaining severity-dependent.

## 3. Problem Formulation

Let building `i` provide an hourly electricity time series `y_i,t`. For each forecast origin `t`, the input `x_i,t` contains a fixed-length window of lagged load values and available covariates such as calendar features. The task is next-hour point forecasting:

```text
y_hat_i,t+1 = f(x_i,t).
```

The evaluation is leave-one-target-building in spirit. For a target building `i*`, all other buildings in the benchmark are treated as candidate source buildings. Source buildings provide training windows from their chronological training periods. The target building provides only the first `k` days of its own training period for adaptation, where `k` is in `{3, 7, 14, 30}`. The target validation and test periods occur later in time.

The central performance metric is test-period MAE:

```text
MAE_i = mean_t |y_i,t+1 - y_hat_i,t+1|.
```

A method is considered useful in this cold-start setting only if it improves over strong operational baselines under the same information budget. In particular, the persistence baseline is:

```text
y_hat_i,t+1 = y_i,t.
```

This baseline defines the residual that the proposed methods attempt to learn. The target test period is never used for source selection, similarity computation, residual scaling, calibration, model fitting, or hyperparameter selection. Repeated seeds are treated as repeated runs and are averaged before building-level comparisons.

## 4. Materials and Methods: Persistence-Anchored Multi-Scale Residual Transfer

### 4.1 Design Rationale

The proposed method targets a specific failure mode in cold-start cross-building forecasting: direct transfer of absolute load values is sensitive to building scale, activity level, and source-target mismatch. The method therefore changes the learning target before changing the model class. Rather than training a regressor to output the next absolute load, it trains the model to output the change that persistence does not explain. This distinction matters because persistence is not a separately trained predictor to be ensemble-weighted. It is the reference point that defines the supervised target.

The framework has three core components:

1. a persistence-anchored residual target;
2. residual RevIN, which normalizes each residual window to reduce instance-level distribution shift;
3. multi-scale residual encoding, which summarizes short-, medium-, and daily-scale residual dynamics.

The default reported variant is denoted PA-MSR. An enhanced variant, PA-MSR+, additionally uses daily-profile source selection and target-side calibration. PA-MSR+ is evaluated as a conditional extension rather than as a guaranteed improvement, because profile-based source selection is unreliable when the target building has too few adaptation days. The implementation uses histogram gradient boosting as the residual learner because tree ensembles are strong small-data baselines in these experiments. The residual formulation itself is model-agnostic.

### 4.2 Persistence-Anchored Residual Target

For an input window ending at time `t`, let `y_t` be the most recent observed load and `y_t+1` be the next-hour target. A direct source-target model learns:

```text
y_t+1 = f_theta(x_t).
```

PA-MSR instead learns the one-step residual:

```text
r_t+1 = y_t+1 - y_t.
```

Because residual magnitudes differ across buildings, the residual target is normalized by a robust target-specific residual scale. Let `s_i` denote the residual scale for building `i`, computed from the allowed training or few-shot adaptation period:

```text
z_i,t+1 = (y_i,t+1 - y_i,t) / s_i.
```

The residual scale is estimated from first differences using a robust rule based on the larger of the IQR-based standard deviation estimate, the empirical standard deviation, and a small positive floor. The model is trained to predict `z_i,t+1`, not `y_i,t+1`. At test time, the forecast is reconstructed as:

```text
y_hat_i,t+1 = y_i,t + alpha_i * s_i * f_theta(phi_i(x_i,t)),
```

where `phi_i(x_i,t)` denotes the normalized tabular feature representation and `s_i` maps the predicted standardized residual back to the target load scale. In PA-MSR, `alpha_i` is fixed to 1, so the full learned residual correction is used. In PA-MSR+, `alpha_i` is selected on the validation period to calibrate the residual amplitude. If `alpha_i = 0`, the method reduces exactly to persistence.

This design differs from a post-hoc weighted ensemble of persistence and a second predictor. In an ensemble, both components usually predict the same absolute target and are combined after training. In PA-MSR, the supervised target itself is the residual around persistence, so the learner is never asked to reproduce the dominant persistence component. When PA-MSR+ is used, the calibration factor controls only the magnitude of the learned residual correction.

### 4.3 Residual RevIN and Multi-Scale Residual Encoding

Residual RevIN is used to reduce instance-level scale and offset differences before residual learning, following the broader motivation of reversible instance normalization for time-series distribution shift [14]. For each input window, the energy channel is normalized by window-specific statistics, and the predicted residual is mapped back through the corresponding residual scale. This differs from global source-fitted scaling because it does not force a target building to share the absolute load scale of source buildings.

Multi-scale residual encoding augments the flattened window representation with residual summaries over 3-, 6-, 12-, and 24-hour spans. For each span, the feature set includes the deviation of the last value from the span mean, the span standard deviation, the mean first difference, and the mean absolute first difference. These features encode short- and daily-scale residual structure without introducing a heavy neural backbone. The resulting default variant is:

```text
PA-MSR = persistence anchor + residual RevIN + multi-scale residual encoding.
```

### 4.4 Daily-Profile Source Selection and Calibration

The enhanced variant PA-MSR+ adds two conditional refinements. Daily-profile source selection computes a 24-hour residual profile from source training data and from the target few-shot adaptation window. It then prioritizes source buildings with similar residual profile shapes. Target-side calibration chooses a scalar residual amplitude on the validation period. These refinements require enough target data to estimate profile and calibration statistics, so they are expected to help more at k=14 and k=30 than at k=3.

```text
PA-MSR+ = PA-MSR + daily-profile source selection + validation calibration.
```

### 4.5 Feature Representation

Each time window is converted to a tabular representation for tree-based regression. The energy channel is normalized using the building-specific mean and robust scale computed from the allowed training data. The feature vector includes the flattened normalized window, first differences within the window, and window-level summary statistics. These statistics include the most recent normalized load, mean, standard deviation, minimum, maximum, latest difference, and mean absolute difference. Calendar and other available covariates from the original forecasting feature set are retained when present.

The same feature construction is applied to source, target adaptation, validation, and test windows. In few-shot settings, target-building normalization statistics are computed only from the allowed target adaptation period. Test-period values are never used to fit normalizers, source weights, calibration factors, or model parameters.

### 4.6 Similarity-Aware Source Weighting

For each target building, the residual learner trains on both source samples and few-shot target samples. In PA-MSR+, source samples are not treated equally. A source-target signature is computed from source training data and from the target few-shot adaptation data. Earlier residual boosting experiments used normalized load scale and variability, zero-load ratio, percentile range, mean absolute first difference, and a 24-hour mean load profile. PA-MSR+ shifts this idea toward daily residual profiles, because the module is intended to select sources with similar residual dynamics rather than similar absolute load levels. These signatures are computed without access to the target test period.

Let `q_i` be the signature of source building `i`, and let `q_*` be the signature of the target building. Distances are computed after standardizing signature dimensions over the source set:

```text
d_i = || standardize(q_i) - standardize(q_*) ||_2.
```

The source-building weight is then computed with a softmax kernel:

```text
w_i = exp(-d_i / tau) / sum_j exp(-d_j / tau).
```

In the reported experiments, only the nearest source buildings are retained when a top-k source constraint is used. Source samples from buildings with higher zero-load ratios are also down-weighted by a reliability factor `1 / (1 + zero_ratio_i)`. The resulting sample weights are normalized to have mean one before model fitting. Few-shot target samples are included with an explicit target weight, so the target adaptation data remain influential even when many source windows are available.

### 4.7 Target-Side Residual Calibration

After fitting the residual model, PA-MSR+ selects the residual amplitude `alpha_i` using the target validation period. For a small grid of candidate values, the method evaluates:

```text
y_hat_i,t+1(alpha) = y_i,t + alpha * s_i * f_theta(phi_i(x_i,t)).
```

The value with the lowest validation MAE is used for the test period. The grid used in the implementation is:

```text
{0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25}.
```

This calibration is intentionally low-dimensional. It does not tune the structure of the boosting model on the test set, and it does not choose between independently trained predictors. Its role is to shrink or expand the residual correction when the target validation period indicates that the transferred residual dynamics are too weak or too aggressive.

### 4.8 Training and Prediction Procedure

For a target building and a target-data budget `k`, PA-MSR and PA-MSR+ follow the same chronological data restrictions as all baselines:

1. Split each building chronologically into training, validation, and test periods.
2. Use only the first `k` days of the target training period for target adaptation.
3. Compute target normalization, residual scale, and target signature from the target few-shot data.
4. Compute source signatures from source training periods and assign source weights by similarity.
5. Build residual targets for all source and target adaptation windows.
6. Fit a weighted residual regressor on source residual windows plus up-weighted target residual windows.
7. Set `alpha_i = 1` for PA-MSR; for PA-MSR+, select `alpha_i` on the target validation period.
8. Evaluate once on the target test period.

This procedure ensures that the test period is used only for final evaluation. Source selection, residual scaling, sample weighting, and residual calibration are all determined before test predictions are scored.

## 5. Experimental Setup and Evaluation Protocol

### 5.1 Research Questions

The experiments are organized around four research questions:

RQ1 asks whether persistence and source-target tree ensembles are strong enough to change the interpretation of cold-start transfer learning. RQ2 asks whether persistence-anchored residual transfer improves the BDG2 cold-start benchmark under the same chronological protocol. RQ3 asks whether residual RevIN and multi-scale residual encoding add value beyond the persistence anchor. RQ4 asks whether the method generalizes to an external active-building dataset and whether PA-MSR+ should be treated as a conditional enhancement rather than a universal full model.

### 5.2 Datasets and Benchmark Roles

We use BDG2 and COFACTOR as complementary evaluation regimes rather than as interchangeable leaderboards.

The main benchmark is a 24-building BDG2 cold-start benchmark. BDG2 is an open non-residential building energy meter dataset derived from the ASHRAE Great Energy Predictor III competition and related Building Data Genome releases [6]. Each building is used as the target once, and the remaining selected buildings are treated as sources. This benchmark is used for the full method matrix, including persistence, source-target tree ensembles, neural transfer baselines, residual neural variants, prior PARBoost variants, PA-MSR, and PA-MSR+. It is also used for failure-mode analysis because it contains both active and persistence-dominated target cases.

To address the concern that the main benchmark may be too small, we construct a 120-building active BDG2 robustness subset from the full eligible pool. The eligibility audit starts from 1578 BDG2 buildings and retains buildings with sufficient chronological coverage and active test-period behavior. The robustness subset is stratified across building types where possible. We use this subset for a PA-MSR-only scale validation against persistence, seasonal naive, source-target tree ensembles, and the earlier residual boosting baseline. It is not a full neural-method matrix and should not be interpreted as an exhaustive evaluation on every eligible BDG2 building.

External validation uses 44 active buildings from the COFACTOR Drammen dataset, which provides multi-year hourly energy and contextual data for 45 public buildings in Drammen, Norway [12,13]. One building is excluded by the benchmark coverage and activity filters before model evaluation. COFACTOR is used to test whether the persistence-anchored residual idea remains useful outside BDG2. Because the retained COFACTOR targets are active buildings and source-target tree ensembles perform very well, it is treated as an external validation setting against strong baselines rather than as a weak-baseline demonstration.

**Table 1. Dataset roles and comparison scope.** The three benchmarks are used for complementary claims rather than merged into a single pooled leaderboard.

| Benchmark | Target buildings | Target status | Main role | Method scope | Supported claim boundary |
|---|---:|---|---|---|---|
| BDG2-24 | 24 | 17 active, 7 inactive or near-zero | Main controlled benchmark and failure-mode analysis | Full method matrix including neural transfer baselines, tree baselines, residual variants, PA-MSR, and PA-MSR+ | Supports detailed method comparison and negative-transfer diagnosis on a controlled subset |
| BDG2-120 active | 120 | Active test periods | Larger-sample robustness check | PA-MSR against persistence, seasonal naive, source-target tree ensembles, and prior residual boosting | Supports PA-MSR scale robustness on active buildings, but not a full neural matrix or all-eligible BDG2 claim |
| COFACTOR-44 | 44 | Active public buildings | External validation outside BDG2 | PA-MSR and PA-MSR+ against strong simple and tree baselines | Supports external active-building validation and severity-aware use of PA-MSR+ |

### 5.3 Cold-Start Split Protocol

All datasets use the same chronological evaluation structure. Each target building is split into 60% training, 20% validation, and 20% test periods. Few-shot adaptation uses only the first `k` days of the target training period, with `k` in `{3, 7, 14, 30}`. The target building is excluded from the source set. Source-target methods may use source-building training windows and the permitted target few-shot windows, but not target validation or test windows for fitting the forecasting model.

The validation period is used only for model selection, early stopping where applicable, and residual-amplitude calibration in calibrated variants. It is not used to fit the residual regressor or any baseline forecasting model, and it is not included in the few-shot adaptation budget. To separate the effect of validation-based calibration from the main residual formulation, the ablation study includes residual variants without target-side calibration. The target test period is used only once, after all source selection, normalization, calibration, and model fitting decisions have been made.

### 5.4 Baselines and Comparison Scope

The baseline set is chosen to prevent weak-baseline conclusions. Persistence predicts the next-hour load as the last observed load. Seasonal naive predicts the load from the same hour in the previous day. Source-target classical baselines include Random Forest, ExtraTrees, and histogram gradient boosting trained with the same allowed source and target data budget [8-10]. The 24-building BDG2 benchmark also includes neural transfer baselines, including DLinear target-only, DLinear pretrain-and-fine-tune, DLinear source replay fine-tuning, and residual neural PA-SRFT variants where available [7].

The comparison scope differs by benchmark for computational and evidential reasons. BDG2-24 provides the full method matrix and is used to compare the proposed residual-transfer variants with neural transfer methods. BDG2-120 provides a PA-MSR-only scale validation against strong classical baselines and the earlier residual boosting baseline. COFACTOR-44 is used for external validation of PA-MSR and PA-MSR+ against the strongest simple/classical baselines. Therefore, claims about neural-transfer failure are made from the BDG2-24 full matrix, claims about large active-building robustness are made from BDG2-120, and claims about external generalization are made from the COFACTOR-44 paired comparisons.

### 5.5 Residual-Transfer Module Audit

The ablation study has two levels. The first tests whether the earlier persistence-anchored residual boosting baseline is more than a tuned tree model. It compares five variants under the same split and feature construction:

- direct HistGBDT source-target prediction of absolute load;
- residual prediction with uniform source weights;
- residual prediction with source-target similarity weighting;
- residual prediction with target-side calibration;
- full earlier residual boosting with residual prediction, similarity weighting, and calibration.

This ablation isolates the effect of changing the prediction target from absolute load to persistence-anchored residuals. It also tests whether similarity weighting and calibration provide additional benefits beyond the residual target itself.

The second ablation tests the newer module stack:

- `M1_ANCHOR`: persistence-anchored residual learning;
- `M3_ANCHOR_REVIN`: residual RevIN;
- `M9_REVIN_MSR`: residual RevIN plus multi-scale residual encoding;
- `M11_REVIN_MSR_DPS_CAL`: PA-MSR plus daily-profile source selection and validation calibration.

This second ablation is used for the final method claim. In the paper-facing tables, `M9_REVIN_MSR` is reported as PA-MSR and `M11_REVIN_MSR_DPS_CAL` is reported as PA-MSR+. PARBoost denotes the earlier residual boosting baseline only, not the final proposed method. This naming distinction is important because the final claim is based on PA-MSR and PA-MSR+, while PARBoost is retained as historical and supplementary evidence for the persistence-anchored residual principle.

### 5.6 Metrics, Ranking, and Statistical Testing

The primary forecasting metric is MAE because the task is next-hour point forecasting and MAE is less dominated by rare large errors than squared-error metrics. We also report RMSE and sMAPE in supplementary building-level outputs. Main-paper summaries use mean MAE, median MAE, P90/P95 MAE, worst-case MAE, mean rank, and top-3 rate. Tail metrics are included because cold-start transfer methods can appear competitive on average while producing unacceptable failures on individual buildings.

For each building, k value, and method, repeated seeds are averaged before method comparison. The building, not the seed, is the statistical unit of generalization. Paired comparisons are computed at the building level using MAE deltas between PA-MSR or PA-MSR+ and each baseline. We report win rate, mean and median paired delta, bootstrap confidence intervals over buildings, and Wilcoxon signed-rank tests. Seeds are not treated as independent samples.

Negative transfer is reported as a method performing worse than a specified baseline, usually persistence or a strong source-target tree ensemble, on the same target building and k value. This definition is intentionally baseline-dependent: failure relative to persistence indicates that the transfer method does not justify replacing the simplest operational predictor, whereas failure relative to RF-ST or ExtraTrees-ST indicates that the proposed transfer mechanism does not beat strong classical source-target learning.

### 5.7 Reproducibility and Leakage Controls

All comparisons in the main tables are produced under the same chronological split, target-data budgets, feature construction, and evaluation metrics. Source-target similarity and daily-profile matching are computed only from source training data and the target few-shot adaptation period. Target normalization and residual scales are computed only from allowed target data. The target test period is never used for source selection, similarity computation, model fitting, calibration, hyperparameter selection, or threshold selection.

Per-building predictions, metrics, building manifests, validation summaries, paired-test tables, and figure source data are saved to disk. The paper-facing evidence tables are generated from building-level validation outputs so that main-paper claims can be traced back to per-target results rather than only aggregate leaderboards.

## 6. Results

### 6.1 Strong Baselines Are Not Optional in Cold-Start Evaluation

The first result is that cold-start building forecasting is already a difficult benchmark before transfer learning is considered. On the 24-building BDG2 benchmark, persistence is a strong next-hour predictor, with a mean MAE of 2.938 and a median MAE of 0.429 across target-data budgets. Source-target tree ensembles are also competitive, especially when target data are scarce. Improvement over target-only neural models is therefore not sufficient evidence for a useful cold-start method. The method must also be evaluated against persistence and classical source-target ensembles under the same split.

The neural transfer baselines further show why robustness, not only average accuracy, is central in this setting. DLinear-SRFT improves over weaker target-only neural variants on some buildings, but it also produces severe failures. On the 24-building BDG2 benchmark, its mean MAE is 12.423 at k=3 and its worst-case MAE reaches 154.119. These failures motivate the central design choice of the residual-transfer framework. The model should not directly transfer absolute load dynamics across heterogeneous buildings. It should learn residual corrections around a strong persistence anchor.

### 6.2 PA-MSR and PA-MSR+ Improve the BDG2 Main Benchmark

On the 24-building BDG2 benchmark, the multi-scale residual variants improve over the earlier residual boosting baseline and over the strongest source-target tree ensembles (Figure 2a). PA-MSR obtains the lowest mean MAE at k=3 and k=30, with mean MAE values of 2.613 and 2.000. PA-MSR+ obtains the lowest mean MAE at k=7 and k=14, with values of 2.387 and 2.243. Both variants remain competitive in median and worst-case MAE. The median values must nevertheless be interpreted carefully because seven of the 24 BDG2 targets have inactive or near-zero test periods.

**Table 2. BDG2-24 main cold-start benchmark.** Values are building-level summaries after averaging repeated seeds. PA-MSR corresponds to `M9_REVIN_MSR`; PA-MSR+ corresponds to `M11_REVIN_MSR_DPS_CAL`.

| k | Best method by mean MAE | Mean MAE | Median MAE | Worst MAE | Strongest tree baseline mean MAE | Persistence mean MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | PA-MSR | 2.613 | 0.296 | 37.27 | 3.218 | 2.938 |
| 7 | PA-MSR+ | 2.387 | 0.390 | 30.21 | 2.923 | 2.938 |
| 14 | PA-MSR+ | 2.243 | 0.375 | 27.94 | 2.830 | 2.938 |
| 30 | PA-MSR | 2.000 | 0.264 | 25.38 | 2.708 | 2.938 |

Because BDG2-24 includes seven inactive or near-zero targets, we also report the active-target subset (Figure 4). The same method pattern remains visible when those near-zero cases are removed: PA-MSR is strongest at k=3/30, and PA-MSR+ is strongest at k=7/14. The active subset has larger absolute MAE because it excludes targets that persistence predicts almost perfectly.

**Table 3. BDG2-24 active-target sensitivity.** The table uses only the 17 active targets.

| k | Best residual variant on active targets | Mean MAE | Median MAE | P90 MAE | Residual RevIN mean MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | PA-MSR | 3.689 | 0.605 | 6.935 | 3.827 |
| 7 | PA-MSR+ | 3.370 | 0.691 | 7.261 | 3.667 |
| 14 | PA-MSR+ | 3.167 | 0.691 | 7.206 | 3.385 |
| 30 | PA-MSR | 2.824 | 0.482 | 6.218 | 2.926 |

The paired comparisons are more informative than the aggregate leaderboard. PA-MSR significantly improves over the RevIN-only residual model (`M3_ANCHOR_REVIN`) at all four k values, with win rates of 79.2-87.5% and Wilcoxon p values below 0.003. Against Random Forest source-target and ExtraTrees source-target, PA-MSR wins on at least 95.8% of buildings for k=7 and on all buildings for k=3, 14, and 30, except for one k=7 comparison. Against the previous residual boosting variant, PA-MSR reduces mean MAE at every k, but the win rate is only 54.2-62.5%. The supported claim is therefore that PA-MSR improves the residual-transfer mechanism and the mean leaderboard, not that it eliminates all building-level negative transfer.

### 6.3 BDG2-120 Supports PA-MSR Scale Robustness

To test whether PA-MSR is merely an artifact of the 24-building benchmark, we additionally evaluate PA-MSR on a 120-building active BDG2 robustness subset. This subset has larger and more variable loads, so its absolute MAE values should not be compared directly with the 24-building benchmark. Within this robustness protocol, PA-MSR remains the top-ranked method by mean MAE and mean rank across all k values (Supplementary Figure S1). Its mean MAE decreases from 16.52 at k=3 to 15.84 at k=30, and its mean ranks remain close to first place, ranging from 1.308 to 1.467.

The building-paired comparisons provide the main evidence for scale robustness. PA-MSR wins against persistence on 95.8%, 96.7%, 97.5%, and 97.5% of buildings as k increases from 3 to 30 days, with negative-transfer rates of only 4.2%, 3.3%, 2.5%, and 2.5%. It also improves over Random Forest source-target with win rates of 92.5-95.0% and over ExtraTrees source-target with win rates of 91.7-95.8%. Against the earlier residual boosting baseline, PA-MSR wins on 80.8-89.2% of buildings, although the k=7 negative-transfer rate remains 19.2%. The supported claim is therefore that PA-MSR scales well on a larger active-building subset and improves the previous residual boosting baseline, not that all building-level failures are eliminated.

**Table 4. BDG2-120 active robustness validation.** PA-MSR is evaluated on the larger active subset against the existing BDG2-120 baselines. Values are building-level summaries for 120 target buildings.

| k | Best method by mean MAE | Mean MAE | Median MAE | P90 MAE | P95 MAE | Worst MAE | Mean rank | Top-3 rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | PA-MSR | 16.52 | 10.47 | 40.71 | 47.01 | 95.65 | 1.308 | 98.3% |
| 7 | PA-MSR | 16.43 | 10.82 | 40.31 | 45.70 | 92.87 | 1.467 | 95.0% |
| 14 | PA-MSR | 16.20 | 10.80 | 39.61 | 44.00 | 88.33 | 1.375 | 98.3% |
| 30 | PA-MSR | 15.84 | 10.13 | 37.04 | 43.09 | 87.29 | 1.408 | 95.8% |

### 6.4 COFACTOR Supports External Active-Building Validation

The earlier residual boosting result on COFACTOR was persistence-safe but did not outperform ExtraTrees-ST. The multi-scale residual variants change this conclusion (Figure 2b). On the 44 active COFACTOR buildings, PA-MSR is the strongest method at k=3 and k=7, with mean MAE values of 3.518 and 3.633. PA-MSR+ is strongest at k=14 and k=30, with mean MAE values of 3.601 and 3.523. Both variants outperform persistence, Random Forest source-target, ExtraTrees source-target, HistGBDT source-target, and the earlier residual boosting variant in the aggregate leaderboard.

**Table 5. COFACTOR-44 external active-building validation.** COFACTOR-44 is used to test whether the final multi-scale residual variants generalize outside BDG2 under the same cold-start protocol.

| k | Best method by mean MAE | Mean MAE | Median MAE | Worst MAE | ExtraTrees-ST mean MAE | Persistence mean MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | PA-MSR | 3.518 | 2.788 | 10.34 | 3.859 | 4.395 |
| 7 | PA-MSR | 3.633 | 2.889 | 10.72 | 3.858 | 4.395 |
| 14 | PA-MSR+ | 3.601 | 2.807 | 10.37 | 3.771 | 4.395 |
| 30 | PA-MSR+ | 3.523 | 2.763 | 9.953 | 3.706 | 4.395 |

The paired results support this external validation but also clarify the severity dependence of the modules (Figure 3). PA-MSR beats persistence on all COFACTOR buildings at all k values and beats ExtraTrees-ST with win rates of 97.7%, 75.0%, 65.9%, and 88.6% for k=3, 7, 14, and 30. PA-MSR+ is weaker than PA-MSR at k=3 and k=7, but becomes strongest at k=14 and k=30. This pattern supports a conditional interpretation: daily-profile selection and calibration should be used when the target adaptation window is long enough, not as a default extreme-cold-start correction.

### 6.5 RevIN and MSR Are the Supported Core Modules

The ablation study identifies residual anchoring as the first necessary step, but the final module analysis shows that residual anchoring alone is not sufficient. A direct source-target HistGBDT model that predicts absolute load fails badly on BDG2, with mean MAE values around 17-19 across k values. When the same modeling family is reformulated to predict residual corrections around persistence, mean MAE drops to approximately 2-3. Residual RevIN then provides a consistent improvement over the anchor-only residual model on COFACTOR-44, with building-level win rates of 95-100% across k values.

MSR is the second supported core module, but its evidence differs by dataset. On BDG2-24, PA-MSR improves over `M3_ANCHOR_REVIN` at every k, with win rates of 79.2-87.5% and significant paired tests. In the COFACTOR module screen, MSR also improves over the RevIN-only residual model, while the full COFACTOR-44 run shows smaller paired gains over `M3_ANCHOR_REVIN` at k=3/7/14 and clearer gains at k=30. This distinction matters: COFACTOR-44 supports external leaderboard generalization of PA-MSR and PA-MSR+, but the strongest module-level evidence for MSR comes from BDG2-24 and the COFACTOR module screen. By contrast, ordinary similarity weighting is not reliable, and residual gates degraded performance in pilot experiments. The mechanism claim is therefore specific: persistence anchoring, residual RevIN, and MSR are core modules; daily-profile selection and calibration are conditional refinements.

**Table 6. Building-paired robustness evidence for PA-MSR.** Positive improvement means lower MAE than the baseline. The building, not the seed, is the statistical unit.

| Dataset | Baseline | k | Mean improvement | Win rate | Negative-transfer rate |
| --- | --- | ---: | ---: | ---: | ---: |
| BDG2-24 | Residual RevIN | 3 | 0.098 | 79.2% | 16.7% |
| BDG2-24 | Residual RevIN | 7 | 0.077 | 83.3% | 16.7% |
| BDG2-24 | Residual RevIN | 14 | 0.098 | 83.3% | 16.7% |
| BDG2-24 | Residual RevIN | 30 | 0.072 | 87.5% | 12.5% |
| BDG2-120 | Persistence | 3 | 2.961 | 95.8% | 4.2% |
| BDG2-120 | Persistence | 7 | 3.055 | 96.7% | 3.3% |
| BDG2-120 | Persistence | 14 | 3.282 | 97.5% | 2.5% |
| BDG2-120 | Persistence | 30 | 3.644 | 97.5% | 2.5% |
| BDG2-120 | ExtraTrees-ST | 3 | 6.578 | 95.8% | 4.2% |
| BDG2-120 | ExtraTrees-ST | 7 | 5.768 | 91.7% | 8.3% |
| BDG2-120 | ExtraTrees-ST | 14 | 5.778 | 91.7% | 8.3% |
| BDG2-120 | ExtraTrees-ST | 30 | 5.755 | 92.5% | 7.5% |
| BDG2-120 | Prior residual boosting | 3 | 1.796 | 89.2% | 10.8% |
| BDG2-120 | Prior residual boosting | 7 | 1.010 | 80.8% | 19.2% |
| BDG2-120 | Prior residual boosting | 14 | 0.908 | 85.0% | 15.0% |
| BDG2-120 | Prior residual boosting | 30 | 0.847 | 85.8% | 14.2% |
| COFACTOR-44 | ExtraTrees-ST | 3 | 0.341 | 97.7% | 2.3% |
| COFACTOR-44 | ExtraTrees-ST | 7 | 0.225 | 75.0% | 25.0% |
| COFACTOR-44 | ExtraTrees-ST | 14 | 0.138 | 65.9% | 34.1% |
| COFACTOR-44 | ExtraTrees-ST | 30 | 0.172 | 88.6% | 11.4% |

### 6.6 Gains Depend on the Persistence Baseline and Target-Data Budget

The paired results clarify when residual correction has room to improve over the anchor. On BDG2-120, PA-MSR's mean paired gain over persistence rises from 2.961 at k=3 to 3.644 at k=30, while negative-transfer rates remain below 5% against persistence at every k. This pattern is consistent with the residual-transfer design. When persistence is already nearly optimal, the residual correction has limited room to help. When persistence leaves systematic residual error and more target data are available, source-informed residual learning becomes more useful.

COFACTOR shows the same qualitative pattern at a smaller absolute scale. The multi-scale variants improve over persistence in all k values, but the magnitude of the gain depends on the target-data budget and on whether PA-MSR or PA-MSR+ is used. This supports a nuanced conclusion: multi-scale residual transfer can outperform strong tree baselines on an external active-building dataset, but the full enhanced variant should not be applied blindly when k is very small.

## 7. Discussion

The results support a baseline-aware view of cold-start building forecasting. A method that improves over target-only neural training may still be weak if it cannot beat persistence or source-target tree ensembles. This matters because persistence is not a trivial baseline in next-hour building electricity forecasting; it captures the strong short-term autocorrelation present in many buildings. Tree ensembles are also strong because they can use lagged load and calendar features efficiently with limited target data. For this reason, cold-start transfer studies should treat persistence, Random Forest, ExtraTrees, and gradient boosting as core baselines rather than secondary checks.

PA-MSR is effective because it changes the transfer problem and then adds only the modules that survive paired testing. Instead of asking a model to learn absolute building load across heterogeneous buildings, it asks the model to learn a normalized residual correction to persistence. Residual RevIN further reduces instance-level distribution shift, and MSR adds explicit short- and daily-scale residual structure. The ablation results show that these design choices are not cosmetic: direct load prediction with the same model family fails badly on BDG2, residual prediction sharply reduces both average and tail error, and MSR improves over the RevIN-only residual model most clearly on BDG2-24 and in the COFACTOR module screen.

The stratified analysis further clarifies the role of the persistence anchor. Residual-transfer methods have little room to improve when persistence already leaves almost no residual error. Their gains become larger when persistence has nontrivial error that can be corrected from source-informed residual patterns. This behavior is desirable for a cold-start method because it avoids forcing transferred corrections onto targets that do not need them, while still allowing source information to help when persistence is insufficient.

The COFACTOR results support external generalization after introducing MSR, but they still argue against overclaiming. PA-MSR and PA-MSR+ outperform the source-target tree baselines in the aggregate COFACTOR leaderboard, yet the pairwise gains are not uniform across all modules and k values. PA-MSR+ is clearly inappropriate as a default extreme-cold-start model because it underperforms PA-MSR at k=3/7. A defensible interpretation is that multi-scale residual transfer generalizes to an external active-building dataset, while the daily-profile and calibration modules should be applied conditionally.

It is equally important to state what is not claimed. The results do not show that PA-MSR+ is a universally superior full model, that all BDG2 buildings have been exhaustively evaluated with the final PA-MSR matrix, or that improved forecasting MAE solves downstream anomaly detection. They also do not imply that ordinary source-target similarity weighting or residual gates are reliable improvements. These boundaries are part of the contribution: they prevent the method from being interpreted as another weak-baseline transfer result.

These findings also have implications for benchmark construction. Directly combining reported numbers from prior studies would be misleading unless the building subset, cleaning rules, chronological split, horizon, feature set, normalization, and target-data budget are identical. The present study therefore uses a unified protocol for all direct claims and treats prior work as methodological context. This is stricter but more useful: it exposes when an apparent transfer-learning gain disappears under persistence or tree baselines, and it makes negative-transfer cases visible instead of averaging them away.

Finally, the results separate forecasting from anomaly detection. Forecasting residuals are often used as a basis for anomaly scoring, but improved forecasting MAE does not automatically imply better event detection, lower false-alarm rates, or better fault diagnosis. The present paper therefore restricts its main claim to cold-start forecasting. Downstream anomaly detection would require separate labels or controlled anomaly protocols and should be evaluated with detection-specific metrics.

## 8. Limitations

This study has several limitations. First, the main full-method BDG2 benchmark uses a selected 24-building subset rather than every building in the raw dataset. This choice enables controlled comparison with expensive neural baselines and detailed failure analysis. The 120-building active BDG2 robustness subset directly evaluates PA-MSR against persistence, source-target tree ensembles, and the earlier residual boosting baseline, but it does not repeat the full neural method matrix and it is not an exhaustive evaluation on all eligible BDG2 buildings.

Second, COFACTOR is used as an active-building external validation setting rather than as proof of universal superiority. PA-MSR and PA-MSR+ improve the COFACTOR leaderboard relative to persistence and source-target tree ensembles, but the module-level result is severity dependent: PA-MSR is preferable at k=3/7, while PA-MSR+ is preferable at k=14/30. This limits the claim to a severity-aware residual-transfer framework rather than a single full model that is always best.

Third, PA-MSR+ uses a chronological validation period for residual-amplitude calibration. This follows the unified benchmark protocol and does not use the test period, but deployments with only `k` labeled target days and no later validation labels would need a different calibration strategy, such as no calibration, adaptation-period calibration, or a fixed alpha selected from prior deployments.

Fourth, the experiments use lagged load and calendar-derived features. Weather, occupancy, equipment metadata, building control signals, or tariff information may change the relative strength of baselines and transfer methods. The conclusions should therefore be interpreted for the feature setting evaluated here.

Fifth, daily-profile source selection and calibration are not uniformly beneficial at every k value or dataset. The ablation results support persistence anchoring, residual RevIN, and MSR as the core mechanisms, while DPS and calibration should be treated as conditional refinements. Ordinary similarity weighting and residual gates should not be presented as supported contributions.

Finally, this paper addresses forecasting rather than anomaly detection. Although the work was motivated partly by residual-based monitoring, downstream anomaly detection requires separate scoring rules, labels or injection protocols, threshold selection, and detection metrics such as event-level recall, precision, false-alarm rate, and AUPRC.

## 9. Conclusion

This study re-examined transfer learning for cold-start building electricity forecasting under a strict baseline-aware protocol. The results show that persistence and source-target tree ensembles are strong baselines, and that direct neural transfer can produce severe worst-case failures when evaluated against them. These findings challenge evaluations that judge cold-start transfer primarily by improvement over target-only neural models.

We proposed PA-MSR, a persistence-anchored multi-scale residual transfer method that learns normalized residual corrections rather than absolute next-hour load. PA-MSR combines persistence anchoring, residual RevIN, and multi-scale residual encoding. We also evaluated PA-MSR+, an enhanced variant with daily-profile source selection and residual calibration. PA-MSR is the more reliable default under extreme cold start, while PA-MSR+ is useful mainly at k=14/30. A PA-MSR-only validation on the 120-building active BDG2 robustness subset further shows that the method remains top-ranked under a larger active-building sample.

The main conclusion is therefore specific and practical: robust cold-start building forecasting should combine strong baseline evaluation, building-level failure analysis, persistence-anchored residual targets, residual normalization, and multi-scale residual structure. The results do not support an unrestricted "full model is always best" claim. They support a narrower but stronger claim: RevIN and MSR are reliable modules for persistence-anchored cold-start residual transfer, while daily-profile source selection and calibration should be used conditionally.

## Figure Captions

**Figure 1. Cold-start evaluation protocol and no-test-leakage data flow.** Source buildings are used only through their training periods, while the target building contributes only the first `k` days of its training period for few-shot adaptation. The validation period is used for model selection and PA-MSR+ residual-amplitude calibration. The test period is held out until final scoring and is not used for source selection, normalization, fitting, calibration, or hyperparameter selection.

**Figure 2. Main leaderboard across BDG2-24 and COFACTOR-44.** Mean test MAE is reported for the main residual-transfer variants and strong baselines across target adaptation budgets. Panel a shows the BDG2-24 full-method benchmark. Panel b shows the COFACTOR-44 external active-building validation benchmark. PA-MSR corresponds to `M9_REVIN_MSR`, and PA-MSR+ corresponds to `M11_REVIN_MSR_DPS_CAL`.

**Figure 3. Building-paired robustness and conditional PA-MSR+ effect.** Panel a reports PA-MSR paired win rates against persistence, RF-ST, and ExtraTrees-ST on COFACTOR-44. Panel b reports negative-transfer rates for PA-MSR against residual RevIN, persistence, and ExtraTrees-ST on BDG2-24. Panel c compares PA-MSR+ against PA-MSR, showing that PA-MSR+ can underperform under extreme cold start but becomes useful at longer target adaptation windows.

**Figure 4. BDG2 active/inactive sensitivity.** Panel a reports the 17 active BDG2-24 targets. Panel b reports the seven inactive or near-zero targets using a log-scale MAE axis. The active-target panel shows that the PA-MSR/PA-MSR+ pattern is not caused only by near-zero buildings, while the inactive panel documents why median MAE must be interpreted carefully.

## Supplementary Material

Supplementary Table S1 reports the BDG2-120 active robustness leaderboard for PA-MSR, persistence, seasonal naive, RF-ST, ExtraTrees-ST, HistGBDT-ST, and the earlier residual boosting baseline. This table supports scale robustness of PA-MSR on a larger active-building subset, while the full neural matrix remains limited to BDG2-24.

Supplementary Table S2 reports the residual-transfer ablation comparing direct HistGBDT-ST, residual-only learning, residual plus similarity, residual plus calibration, and the earlier full residual boosting baseline. This table motivates persistence anchoring but does not support ordinary similarity as a final core module.

Supplementary Figure S1 reports the BDG2-120 PA-MSR scale-validation leaderboard and paired win rates. Supplementary Figure S2 reports persistence-strength stratification for BDG2-120 and COFACTOR-44. Supplementary Figure S3 reports failure-mode scatter plots for persistence MAE versus neural-transfer degradation on BDG2-24.

Figure source data are stored in `figures/pa_msr/source_data/`. Main evidence tables are stored in `results/final_claim_evidence/`.

## Declarations

### Funding

Funding information should be completed before submission. If no specific funding supported this work, state: "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."

### Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### Data Availability

The BDG2 data used in this study are publicly available through the Building Data Genome Project 2 and the ASHRAE Great Energy Predictor III release [6]. The COFACTOR Drammen dataset is publicly available through Scientific Data and Zenodo [12,13]. Processed manifests, benchmark splits, per-building metrics, final evidence tables, and figure source data are stored in the project repository and should be deposited in a public repository before submission.

### Code Availability

The experiment scripts, evidence-table generation scripts, and figure-generation scripts should be released with the processed benchmark manifests before publication. The current paper-facing figure script is `scripts/make_pa_msr_paper_figures.py`, and the main evidence tables are generated from `results/final_claim_evidence/`.

### CRediT Authorship Contribution Statement

Author contributions should be completed before submission using the CRediT taxonomy. Suggested roles include conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing - original draft, writing - review and editing, visualization, supervision, and project administration.

### Declaration of Generative AI and AI-Assisted Technologies

During manuscript preparation, AI-assisted tools were used to support drafting, editing, code review, and organization of the manuscript. The authors reviewed and edited the content and take full responsibility for the submitted version.

## References

1. K. Amasyali, N.M. El-Gohary, A review of data-driven building energy consumption prediction studies, *Renewable and Sustainable Energy Reviews* 81 (2018) 1192-1205. https://doi.org/10.1016/j.rser.2017.04.095

2. M. Bourdeau, X.Q. Zhai, E. Nefzaoui, X. Guo, P. Chatellier, Modeling and forecasting building energy consumption: A review of data-driven techniques, *Sustainable Cities and Society* 48 (2019) 101533. https://doi.org/10.1016/j.scs.2019.101533

3. M. Ribeiro, K. Grolinger, H.F. ElYamany, W.A. Higashino, M.A.M. Capretz, Transfer learning with seasonal and trend adjustment for cross-building energy forecasting, *Energy and Buildings* 165 (2018) 352-363. https://doi.org/10.1016/j.enbuild.2018.01.034

4. Z. Xing, Y. Pan, Y. Yang, X. Yuan, Y. Liang, Z. Huang, Transfer learning integrating similarity analysis for short-term and long-term building energy consumption prediction, *Applied Energy* 365 (2024) 123276. https://doi.org/10.1016/j.apenergy.2024.123276

5. B. Wei, K. Li, S. Zhou, W. Xue, C. Xu, An instance based multi-source transfer learning strategy for building's short-term electricity loads prediction under sparse data scenarios, *Journal of Building Engineering* 85 (2024) 108713. https://doi.org/10.1016/j.jobe.2024.108713

6. C. Miller, A. Kathirgamanathan, B. Picchetti, P. Arjunan, J.Y. Park, Z. Nagy, P. Raftery, B. Hobson, Z. Shi, F. Meggers, The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition, *Scientific Data* 7 (2020) 368. https://doi.org/10.1038/s41597-020-00712-x

7. A. Zeng, M. Chen, L. Zhang, Q. Xu, Are Transformers Effective for Time Series Forecasting?, *Proceedings of the AAAI Conference on Artificial Intelligence* 37 (2023) 11121-11128. https://doi.org/10.1609/aaai.v37i9.26317

8. L. Breiman, Random Forests, *Machine Learning* 45 (2001) 5-32. https://doi.org/10.1023/A:1010933404324

9. P. Geurts, D. Ernst, L. Wehenkel, Extremely randomized trees, *Machine Learning* 63 (2006) 3-42. https://doi.org/10.1007/s10994-006-6226-1

10. J.H. Friedman, Greedy function approximation: A gradient boosting machine, *The Annals of Statistics* 29 (2001) 1189-1232. https://doi.org/10.1214/aos/1013203451

11. R.J. Hyndman, A.B. Koehler, Another look at measures of forecast accuracy, *International Journal of Forecasting* 22 (2006) 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001

12. S.K. Lien, H.T. Walnum, A.L. Sorensen, COFACTOR Drammen dataset - 4 years of hourly energy use data from 45 public buildings in Drammen, Norway, *Scientific Data* 12 (2025) 393. https://doi.org/10.1038/s41597-025-04708-3

13. A.L. Sorensen, H.T. Walnum, S.K. Lien, COFACTOR Drammen dataset - 4 years of hourly energy use data from 45 public buildings in Drammen, Norway, *Zenodo* (2024). https://doi.org/10.5281/zenodo.11060089

14. T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, J. Choo, Reversible instance normalization for accurate time-series forecasting against distribution shift, *International Conference on Learning Representations* (2022). https://openreview.net/forum?id=cGDAkQo1C0p

15. J. Drgona, J. Arroyo, I.C. Figueroa, D. Blum, K. Arendt, D. Kim, E.P. Olle, J. Oravec, M. Wetter, D.L. Vrabie, L. Helsen, All you need to know about model predictive control for buildings, *Annual Reviews in Control* 50 (2020) 190-232. https://doi.org/10.1016/j.arcontrol.2020.09.001

16. A. Kathirgamanathan, M. De Rosa, E. Mangina, D.P. Finn, Data-driven predictive control for unlocking building energy flexibility: A review, *Renewable and Sustainable Energy Reviews* 135 (2021) 110120. https://doi.org/10.1016/j.rser.2020.110120

17. G. Pinto, Z. Wang, A. Roy, T. Hong, A. Capozzoli, Transfer learning for smart buildings: A critical review of algorithms, applications, and future perspectives, *Advances in Applied Energy* 5 (2022) 100084. https://doi.org/10.1016/j.adapen.2022.100084

18. H. Kazmi, J. Suykens, J. Driesen, Transfer learning in demand response: A review of algorithms for data-efficient modelling and control, *Energy and AI* 7 (2022) 100126. https://doi.org/10.1016/j.egyai.2021.100126

19. H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, W. Zhang, Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, *Proceedings of the AAAI Conference on Artificial Intelligence* 35 (2021) 11106-11115. https://doi.org/10.1609/aaai.v35i12.17325

20. H. Wu, J. Xu, J. Wang, M. Long, Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting, *Advances in Neural Information Processing Systems* 34 (2021) 22419-22430. https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html

21. Y. Nie, N.H. Nguyen, P. Sinthong, J. Kalagnanam, A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, *International Conference on Learning Representations* (2023). https://openreview.net/forum?id=Jbdc0vTOcol

22. Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, M. Long, iTransformer: Inverted Transformers Are Effective for Time Series Forecasting, *International Conference on Learning Representations* (2024). https://openreview.net/forum?id=JePfAI8fah

## Reference Use Notes

The references above are used to support field context, method families, datasets, and evaluation choices. Published metric values from prior papers are not imported into the main leaderboard because direct comparison would require identical data cleaning, building subsets, chronological splits, forecast horizons, feature sets, scaling policies, and metrics.
