# Experiment Design: Few-Shot Building Electricity Anomaly Detection via Transfer Learning and Temporal Forecasting

> **Document version**: v2.0 — SCI Q2 target  
> **Date**: 2026-05-10  
> **Status**: Draft — ready for review before implementation  
> **Target venues**: Energy and Buildings (IF~6.7), Building and Environment (IF~7.1), Journal of Building Engineering (IF~6.4), Sustainable Cities and Society (IF~10.5), Energy (IF~9.0)

---

## 1. Paper Positioning

- **Paper type**: New Setting / New Problem
- **Core story**: No anomaly labels + scarce target-building data → pretrain on source buildings → few-shot fine-tune → stable forecasts → reliable anomaly scoring
- **Main claim**: Transfer learning improves **forecast stability** (not accuracy), which in turn makes anomaly scoring more reliable for cold-start buildings.
- **Differentiation**: We do NOT claim to build a better forecasting model. We claim that transfer-learned forecasts, even if not the most accurate, produce **more stable residuals** — and stable residuals are what make anomaly detection work in cold-start settings.

---

## 2. Research Problem

**Formal definition**:

Given:
- A set of source buildings $\mathcal{S} = \{B_1, \dots, B_m\}$ with at least 1 year of hourly electricity data,
- A target building $B_T$ with **zero anomaly labels** and only $k$ days of normal-operation data ($k \in \{1, 3, 7, 14\}$),
- A forecasting model $f_\theta$ that predicts $\hat{y}_{t+1}$ from the past $W$ hours,

**Goal**: Produce an anomaly score $s_t$ for each timestamp such that $\text{AUROC}(s, y_{\text{anom}}) \gg 0.5$ and false-alarm rate is controlled, despite having no labeled anomalies for calibration.

**Hard constraints**:
1. No anomaly labels available for $B_T$.
2. Only $k$ days of $B_T$ normal data for adaptation to the building's load profile.
3. Point-level (hourly) anomaly detection.
4. Chronological data — no shuffling, no future leakage.

---

## 3. Method Overview: TL-TFAD

**TL-TFAD** = Transfer Learning based Temporal Forecasting for Anomaly Detection

```
┌──────────────────────────────────────────────────────────────────────┐
│  Module A: Multi-source Pretraining                                   │
│  Source buildings S → shared forecasting model f_θ                    │
│  Input: [y_{t-W:t}, hour, dow, is_weekend, month, (sqm)]             │
│  Backbone: LSTM / GRU / TCN / DLinear / PatchTST                      │
├──────────────────────────────────────────────────────────────────────┤
│  Module B: Few-shot Fine-tuning                                       │
│  k ∈ {0, 1, 3, 7, 14} days of B_T → adapt f_θ                        │
│  Strategies: full FT / freeze-backbone + FT-head / linear probe       │
│  Early stopping + L2 regularization to prevent overfitting            │
├──────────────────────────────────────────────────────────────────────┤
│  Module C: Stability-Aware Anomaly Scoring                            │
│  e_t = |y_t - ŷ_t|                                                    │
│  rolling_median_t = median({e_{t-W_mad:t}})                           │
│  rolling_MAD_t = median(|e - rolling_median_t|)                       │
│  s_t = (e_t - rolling_median_t) / (rolling_MAD_t + ε)                 │
│  s_t > τ → anomaly, where τ is tuned per-building on validation set  │
└──────────────────────────────────────────────────────────────────────┘
```

**Key design note**: Module B is architecture-dependent. LSTM/GRU/TCN/PatchTST support freeze-then-fine-tune. DLinear is a decomposition + linear layer model — it has no recurrent encoder to freeze. For DLinear, use full fine-tuning and report this architectural distinction. This is not a bug; it's a documented design choice that the paper should explicitly discuss.

---

## 4. Dataset Plan

### 4.1 Data Source

**BDG2** (Building Data Genome 2) — public, hourly electricity for 3,053 buildings over 2 years.

**Backup**: If BDG2 is inaccessible, use ASHRAE Great Energy Predictor III or UCI Appliances Energy dataset. Document the switch decision.

### 4.2 Building Selection (SCI Q2 requirement: ≥ 5 buildings, ≥ 3 types)

**Selection criteria**:
1. ≥ 12 months (8,760 rows) of hourly electricity data.
2. < 15% missing values (stricter than v1.0).
3. At least 3 distinct building use types (e.g., education, office, lodging, retail, healthcare).
4. Within each type, pick the building(s) with the longest complete record.

**Target: 6–8 buildings across 3–4 types**.

| Priority | Type | Minimum count | Rationale |
|---|---|---|---|
| 1 | Office | 2 | Most common; weekday-weekend pattern clear |
| 2 | Education | 2 | Strong seasonal pattern; different from office |
| 3 | Lodging | 2 | 24/7 occupancy; different diurnal pattern |
| 4 | Retail/Other | 1–2 | Optional; increases diversity |

**Candidate buildings** (from prior work — verify existence in downloaded data):
- `Eagle_education_Brianne`
- `Eagle_lodging_Andy`
- `Eagle_office_Mable`

If these don't exist, run a data-quality scan on all BDG2 buildings and pick by criteria above.

### 4.3 Data Files

| File | Description |
|---|---|
| `data/raw/` | Original BDG2 download (untouched) |
| `data/processed/bdg2_electricity_hourly.csv` | Cleaned, merged, feature-engineered |
| `data/processed/building_metadata.csv` | `building_id, sqm, building_type, n_rows, date_range, missing_pct` |
| `data/processed/building_splits.json` | Train/val/test split timestamps per building |

### 4.4 Required Fields

| Field | Description | Required |
|---|---|---|
| `timestamp` | UTC datetime, hourly frequency | **Yes** |
| `building_id` | Unique building identifier | **Yes** |
| `energy` | Hourly electricity consumption (kWh) | **Yes** |
| `hour` | 0–23 | **Yes** (engineered) |
| `day_of_week` | 0–6 (Mon–Sun) | **Yes** (engineered) |
| `is_weekend` | 0/1 | **Yes** (engineered) |
| `month` | 1–12 | **Yes** (engineered) |
| `sqm` | Floor area (m²) | **Nice-to-have** |
| `building_type` | Office/Education/Lodging/Retail/etc. | **Nice-to-have** |

### 4.5 Preprocessing Pipeline

1. **Resample**: Strict hourly frequency. Forward-fill gaps ≤ 2 hours. Gap-fill with linear interpolation for gaps 3–6 hours. Mark gaps > 6 hours as NaN and exclude those segments from training (but keep for test evaluation).
2. **Filter buildings**: Keep buildings with ≥ 12 months data and < 15% NaN.
3. **Engineer time features**: `hour` (sin/cos encoding option), `day_of_week`, `is_weekend`, `month`.
4. **Metadata extraction**: Compute `sqm` and `building_type` if available in BDG2 metadata.
5. **Quality report**: Generate `table_dataset_statistics.csv` — one row per building with `n_rows`, `missing_pct`, `date_range`, `mean_kwh`, `std_kwh`, `sqm`, `building_type`.

---

## 5. Forecasting Task Definition

| Parameter | Default | Alternative values (sensitivity) |
|---|---|---|
| `window_size` | 24 | 12, 48, 168 |
| `horizon` | 1 | — (single-step only) |
| `train/val/test` | 60% / 20% / 20% | Chronological split |
| Normalization | Z-score (fit on train only) | MinMax (compare) |
| Random seeds | 42, 43, 44, 45, 46 | **5 seeds for Q2** |
| Batch size | 64 | 32, 128 |
| Learning rate | 1e-3 | 1e-4, 5e-4, 1e-3, 5e-3 |
| Hidden size (LSTM/GRU) | 128 | 64, 256 |
| Early stopping patience | 10 epochs | — |
| Optimizer | Adam | — |

**Data leakage prohibitions**:
- Split strictly by time — no shuffling.
- Scaler fit ONLY on source-building train set in transfer scenarios.
- Target-building validation set used only for early stopping; no test data seen during training.
- Anomaly injection ONLY on test-set copies; original data never modified.

---

## 6. Baselines (SCI Q2: 3 tiers)

### Tier 1: Classical Statistical Baselines

| Model | Rationale | Implementation |
|---|---|---|
| **ARIMA(p,d,q)** | Standard time-series baseline; order selected via auto_arima or grid search on each building | `statsmodels` |
| **SARIMA** | Captures daily + weekly seasonality; strong baseline for building energy | `statsmodels` |
| **Naive (persistence)** | $\hat{y}_{t+1} = y_t$ — simple sanity check | Manual |

### Tier 2: Deep Learning Forecasting Models

| Model | Rationale | Fine-tune support |
|---|---|---|
| **LSTM** | Workhorse; strong on periodic sequences | Full FT / Freeze+FT / Linear probe |
| **GRU** | Lighter than LSTM; similar performance | Same as LSTM |
| **TCN** | Dilated causal convolution; captures long-range dependencies | Full FT / Freeze+FT |
| **DLinear** (AAAI 2023) | Decomposition + linear layers; SOTA on LTSF benchmarks; simple and reproducible | Full FT only (no encoder structure) |
| **PatchTST** (ICLR 2023) | Patch-based Transformer; strong transfer learning compatibility | Full FT / Freeze+FT |

### Tier 3: Unsupervised Anomaly Detection Baselines (NEW for Q2)

The paper claims to do anomaly detection, so it MUST compare with existing anomaly detection methods.

| Method | Type | How to adapt to cold-start setting |
|---|---|---|
| **Isolation Forest** | Classical unsupervised | Train on source building features (time + energy); test on target |
| **LOF** (Local Outlier Factor) | Density-based | Same as above |
| **LSTM-AE** (Autoencoder) | Deep unsupervised | Train autoencoder on source energy sequences; reconstruction error on target = anomaly score. Fine-tune with k days of target data (comparable to our method). |
| **OmniAnomaly** (AAAI 2019) | VAE + planar NF | Train on source building; optionally fine-tune on k days target. Use reconstruction probability as anomaly score. |

**Implementation note**: LSTM-AE and OmniAnomaly require non-trivial implementation. Priority: LSTM-AE (easier) → OmniAnomaly (if time allows). Isolation Forest and LOF are available in `scikit-learn`. The paper must explain how each baseline is adapted to the cold-start setting — this itself is a methodological contribution.

---

## 7. Experimental Scenarios

### Scenario A: Same-Domain Upper Bound

**Purpose**: Establish the oracle upper bound. "If we had all the data we wanted, how good could forecasting + anomaly scoring be?"

**Setup**:
- Each building: train on first 60%, validate on next 20%, test on last 20%.
- Standard supervised forecasting. No transfer.
- Compute both forecasting metrics AND stability/anomaly metrics (D.1).

**Models**: LSTM, GRU, TCN, DLinear, PatchTST, ARIMA, SARIMA

**Key question**: How close can few-shot transfer get to the upper bound?

---

### Scenario B+C: Transfer + Few-Shot Adaptation (Core Experiment)

**Design**: Direct transfer ($k=0$) and few-shot transfer ($k>0$) are on the same curve.

**Training protocol** (leave-one-building-out cross-validation):

For each target building $B_i$:
1. Source set $\mathcal{S} = \{B_j : j \neq i\}$.
2. Pretrain on $\mathcal{S}$ (all source buildings merged, with `building_id` embedding if architecture supports it).
3. Fine-tune on $k$ days of $B_i$ training data, $k \in \{0, 1, 3, 7, 14\}$.
4. Test on $B_i$'s held-out test set.

**Few-shot data sizes**:
| $k$ (days) | Data points | Ratio (approx.) |
|---|---|---|
| 0 | 0 | 0 (direct transfer / zero-shot) |
| 1 | 24 | ~0.003 |
| 3 | 72 | ~0.01 |
| 7 | 168 | ~0.02 |
| 14 | 336 | ~0.04 |

**Comparison methods**:
| # | Method | Training data | Symbol (in figures) |
|---|---|---|---|
| M1 | **Source-only** | Source buildings only | ◆ |
| M2 | **Target-only few-shot** | k days of target only (train from scratch) | ▲ |
| M3 | **Pretrain + full FT** | Source pretrain + full fine-tune on k days target | ● |
| M4 | **Pretrain + freeze+FT** | Source pretrain + freeze backbone, FT head | ○ |
| M5 | **Same-domain upper bound** | Full target training set | ★ (horizontal dashed line) |

**Source-target combinations**:

With 6 buildings (2 office, 2 education, 2 lodging), three types of transfer:

| Transfer type | Source | Target | Tests |
|---|---|---|---|
| **Cross-type** | Office + Education | Lodging | Tests generalization to unseen building type |
| **Cross-type** | Office + Lodging | Education | Same |
| **Cross-type** | Education + Lodging | Office | Same |
| **Within-type** | Office_A | Office_B | Tests transfer within same type |
| **Within-type** | Education_A | Education_B | Same |
| **Within-type** | Lodging_A | Lodging_B | Same |
| **Many-to-one** | All 5 others | Each of 6 | Full leave-one-out |

Primary results use **many-to-one** (6 groups). Cross-type vs within-type is a sub-analysis for the Discussion.

**Models for this scenario**: LSTM, DLinear, PatchTST (core 3). GRU, TCN (supplementary if resources allow).

**Key output**: 
- `table_few_shot_transfer.csv` — MAE, RMSE, sMAPE, $\sigma_{err}$, $\sigma_{score}$, threshold_variation, clean_FAR for each $k$ × method × model combination.
- **Figure: Few-shot performance curve** — x-axis = $k$ (days), y-axis = metric, one line per method (M1–M5), faceted by model and metric.

---

### Scenario D: Anomaly Detection Evaluation

#### D.1 — Unlabeled Stability Evaluation (PRIMARY)

Applied to ALL scenarios A and B+C.

| Metric | Definition | Why it matters |
|---|---|---|
| **Prediction error std ($\sigma_{err}$)** | $\text{std}(e_t)$ over test set | Core metric: smaller = more stable predictions |
| **Anomaly score std ($\sigma_{score}$)** | $\text{std}(s_t)$ over test set | Smaller = fewer extreme scores on clean data |
| **Threshold variation** | $\text{std}(\tau)$ across rolling windows | Smaller = threshold is reliable |
| **Clean false-alarm tendency** | $\frac{1}{n}\sum \mathbb{I}[s_t > \tau]$ on clean test data | Should be low (~1–5%) |
| **Coefficient of variation of RMSE across seeds** | $\text{CV}_{\text{RMSE}}$ = $\sigma_{\text{RMSE}} / \mu_{\text{RMSE}}$ across 5 seeds | Quantifies training stability |

**Key story**: Even if M3 (pretrain+FT) and M2 (target-only) have similar MAE at some $k$, M3 should show *lower $\sigma_{err}$, $\sigma_{score}$, and CV across seeds*. This is the paper's central empirical claim.

#### D.2 — Controlled Anomaly Injection

**Injection protocol**:
1. Deep-copy test set → `test_injected`.
2. Select anomaly positions randomly (fix seed for reproducibility).
3. Apply anomaly pattern → save `injected_labels.csv` (1 = anomaly, 0 = normal).
4. Run detection pipeline on `test_injected`.
5. Compare detected anomalies vs injected labels.

**Anomaly types (building-realistic)**:

| Type | Real-world cause | Injection method | Parameters |
|---|---|---|---|
| **Point spike** | Sensor glitch, transient load | $y_t \leftarrow y_t + A \cdot \sigma_{\text{local}}$ | $A \sim \mathcal{U}(3, 6)$, duration = 1 |
| **Point drop** | Brief power outage, equipment off | $y_t \leftarrow y_t \cdot f$ | $f \sim \mathcal{U}(0.2, 0.5)$, duration = 1 |
| **Level shift** | Equipment malfunction, HVAC staging failure | $y_{t:t+d} \leftarrow y_{t:t+d} \cdot s$ | $s \sim \mathcal{U}(1.5, 3.0)$, $d \sim \mathcal{U}(3, 12)$ |
| **Gradual drift** | Sensor degradation, slow leak | $y_{t+i} \leftarrow y_{t+i} \cdot (1 + r)^i$ | $r \sim \mathcal{U}(0.005, 0.02)$, $d \sim \mathcal{U}(24, 48)$ |
| **Missing (zero)** | Meter failure, communication loss | $y_{t:t+d} \leftarrow 0$ | $d \sim \mathcal{U}(1, 6)$ |
| **Stuck sensor** | Sensor freeze, repeating value | $y_{t+i} \leftarrow y_{t-1}$ | $d \sim \mathcal{U}(6, 24)$ |
| **Contextual outlier** | Abnormal usage during normal hours | $y_t$ at weekday 10:00 increased by 2× | Specific timestamps (not random) |

**Injection ratios**: 2%, 5% of test points.

**Detection metrics**: Precision, Recall, F1, AUROC, AUPRC, FAR.

**Evaluation note**: Anomaly injection is an **approximation** — real building anomalies are more complex. Position D.2 as: "Since real labels are unavailable, we use controlled injection as a proxy evaluation. Results should be interpreted as evidence of the method's relative ranking, not as absolute detection accuracy claims."

#### D.3 — Comparison with Unsupervised AD Baselines (NEW for Q2)

Apply the same cold-start protocol to the Tier 3 baselines:

| Method | Cold-start adaptation |
|---|---|
| **Isolation Forest** | Train on source building time-feature + energy matrix; score target building |
| **LOF** | Same |
| **LSTM-AE** | Pretrain autoencoder on source buildings; reconstruct target; reconstruction error = anomaly score. Fine-tune on k days target (parallel to our method). |
| **TL-TFAD (ours)** | Pretrain forecaster → fine-tune → rolling MAD score |

**Key output**: `table_anomaly_baseline_comparison.csv` — anomaly detection metrics for each method at each $k$.

**Critical**: This comparison proves your method is *actually better at anomaly detection*, not just better at forecasting. Without this comparison, the paper is "forecasting with transfer learning" — not "anomaly detection."

---

### Scenario E: Ablation Studies

| Ablation | Research question | Variant A | Variant B | Run on |
|---|---|---|---|---|
| **E1: Transfer source** | Does pretraining help at all? | Target-only few-shot (M2) | Pretrain + fine-tune (M3) | All k, all buildings, LSTM + DLinear |
| **E2: Time features** | Do engineered features improve transfer? | energy only (raw sequence) | energy + hour + dow + is_weekend + month | M3, k=7, 3 buildings |
| **E3: Building attributes** | Does sqm information help cross-building transfer? | without sqm in input | with sqm as additional feature | M3, k=7, 3 buildings (skip if sqm unavailable) |
| **E4: Anomaly scoring method** | Is rolling MAD better than static threshold? | Static quantile (95th percentile of val residuals) | Rolling MAD + dynamic τ | M3, k=7, D.1 + D.2 metrics |
| **E5: Fine-tune strategy** (NEW) | Does freezing the backbone matter? | Full FT (M3) | Freeze backbone + FT head (M4) | LSTM only (architecture supports it), k=7 |

**Key output**: `table_ablation.csv`

---

### Scenario F: Robustness and Sensitivity (NEW for Q2)

#### F.1 — Hyperparameter Sensitivity

Vary one parameter at a time; report MAE and $\sigma_{err}$ on 3 buildings with LSTM, M3, k=7.

| Parameter | Values |
|---|---|
| Window size ($W$) | 12, 24, 48, 168 |
| Hidden size | 64, 128, 256 |
| Learning rate | 1e-4, 5e-4, 1e-3, 5e-3 |
| MAD window ($W_{\text{mad}}$) | 12, 24, 48, 168 |
| Anomaly threshold $\tau$ | 2.0, 2.5, 3.0, 3.5, 4.0 |

Present as **line plots** (not a table). Goal: show the method is not brittle to parameter choices.

#### F.2 — Pre-training Data Scale

Vary the number of source buildings used for pretraining.

| # Source buildings | Configuration |
|---|---|
| 1 | Single largest source |
| 2 | Two largest |
| 3 | ...
| All (N−1) | All available |

**Research question**: How many source buildings are enough to learn a useful universal pattern?

This is an **insight contribution** — it tells practitioners how many buildings they need to instrument before the transfer approach becomes worthwhile.

#### F.3 — Noise Robustness

Add Gaussian noise to test data: $\mathcal{N}(0, \eta \cdot \sigma_y)$ with $\eta \in \{0.05, 0.10, 0.20\}$.

Evaluate whether anomaly scoring degrades gracefully with sensor noise.

#### F.4 — Negative Transfer Analysis

Identify cases where M3 (pretrain+FT) performs *worse* than M2 (target-only).

- Report $\Delta$MAE = MAE(M3) − MAE(M2) per building × k combination.
- Analyze: which source-target pairs show negative transfer? Are they cross-type transfers where building usage patterns are fundamentally different?
- Discuss: **when should a practitioner NOT use transfer?**

This transforms a potential weakness into a **scholarly contribution** — the paper not only claims "transfer helps" but also delineates the boundary conditions.

---

## 8. Metrics Summary

### Forecasting Metrics

| Metric | Reported in |
|---|---|
| MAE | All scenarios |
| RMSE | All scenarios |
| sMAPE | All scenarios |
| $R^2$ (coefficient of determination) | Scenario A |

### Stability Metrics (Core Contribution)

| Metric | Reported in | Primary target |
|---|---|---|
| $\sigma_{err}$ (prediction error std) | **Every results table** | Lower = better |
| $\sigma_{score}$ (anomaly score std) | **Every results table** | Lower = better |
| CV of RMSE across seeds | B+C, D.1 | Lower = more training-stable |
| Threshold variation | D.1 | Lower = more reliable |
| Clean false-alarm tendency | D.1 | Lower = better |

### Anomaly Detection Metrics (D.2 + D.3)

| Metric | Reported in |
|---|---|
| Precision, Recall, F1 | D.2, D.3 |
| AUROC, AUPRC | D.2, D.3 |
| False Alarm Rate | D.2, D.3 |
| F1@k (F1 at multiple thresholds) | D.2 (supplementary) |

### Statistical Tests (Across All Scenarios)

| Test | Purpose | Applied to |
|---|---|---|
| **Wilcoxon signed-rank test** | Pairwise comparison of methods | M2 vs M3, M3 vs M4, M3 vs M5, all at k=7 |
| **Diebold-Mariano test** | Forecast accuracy comparison | M2 vs M3 on forecast errors |
| **Cohen's d** | Effect size | Key pairwise comparisons |
| **95% confidence intervals** | Uncertainty quantification | All metrics, reported as mean ± 1.96×SE across 5 seeds and N buildings |

**Reporting format**: "M3 (pretrain+FT) significantly outperforms M2 (target-only) in terms of $\sigma_{err}$ at k=7 (Wilcoxon $p < 0.01$, Cohen's $d = 0.82$)."

---

## 9. Output File Structure

```
results/
├── run_<scenario>_<model>_<building>_<k>_<seed>/
│   ├── config.yaml                # Full run configuration
│   ├── run_manifest.json          # Timestamp, git hash, parameters
│   ├── metrics.json               # All computed metrics
│   ├── predictions.csv            # timestamp, y_true, y_pred, error, anomaly_score
│   ├── anomaly_events.csv         # Detected anomaly timestamps (if applicable)
│   ├── injected_labels.csv        # 0/1 labels (if Scenario D.2)
│   └── figures/
│       ├── prediction_overlay.png
│       └── anomaly_score_timeline.png
│
└── aggregate/
    ├── aggregate_results.csv      # All metrics, all runs, machine-readable
    ├── table_dataset_statistics.csv
    ├── table_same_domain.csv
    ├── table_few_shot_transfer.csv
    ├── table_anomaly_injection.csv
    ├── table_anomaly_baseline_comparison.csv  # NEW: D.3
    ├── table_ablation.csv
    ├── table_negative_transfer.csv            # NEW: F.4
    ├── table_statistical_tests.csv            # NEW: p-values and effect sizes
    └── results_for_paper.md       # Final formatted tables for paper
```

---

## 10. Experiment Execution Plan (SCI Q2 Level)

**Total estimated time: 12–16 weeks** for one person with Python + PyTorch experience.

### Phase 1: Foundation (Weeks 1–3)

| Week | Task |
|---|---|
| 1 | Download BDG2; verify data integrity; select 6–8 buildings; generate `table_dataset_statistics.csv` |
| 2 | Implement preprocessing pipeline → `data/processed/bdg2_electricity_hourly.csv`; build PyTorch Dataset + DataLoader with window construction |
| 3 | Implement LSTM training loop; run Scenario A (LSTM only) on all buildings; validate pipeline end-to-end |

### Phase 2: Core Baselines (Weeks 4–6)

| Week | Task |
|---|---|
| 4 | Implement DLinear; run Scenario A (DLinear) |
| 5 | Implement ARIMA/SARIMA baselines; run Scenario A (classical baselines) |
| 6 | Implement Scenario B+C for LSTM (all k, all buildings, 5 seeds); this is the **core experiment** |

### Phase 3: Transfer + Stability (Weeks 7–9)

| Week | Task |
|---|---|
| 7 | Implement Scenario B+C for DLinear; run all combinations |
| 8 | Implement Module C (rolling MAD scoring); run D.1 stability evaluation on all B+C results |
| 9 | Generate Figure 2 (few-shot performance curves); run Wilcoxon + Diebold-Mariano tests |

### Phase 4: Anomaly Detection (Weeks 10–11)

| Week | Task |
|---|---|
| 10 | Implement anomaly injection (all 7 types, 2 ratios); run D.2 for LSTM + DLinear |
| 11 | Implement LSTM-AE baseline; run D.3 comparison; generate `table_anomaly_baseline_comparison.csv` |

### Phase 5: Ablation + Robustness (Weeks 12–13)

| Week | Task |
|---|---|
| 12 | Run E1–E5 ablations |
| 13 | Run F.1 (hyperparameter sensitivity), F.2 (pretraining scale), F.3 (noise), F.4 (negative transfer) |

### Phase 6: Finalization (Weeks 14–16)

| Week | Task |
|---|---|
| 14 | Implement PatchTST if feasible; re-run key experiments; generate all figures |
| 15 | Generate all aggregate tables; `results_for_paper.md`; case study Figure 3 |
| 16 | Buffer week for re-runs, bug fixes, additional reviewer-oriented experiments |

---

## 11. Core vs. Supplementary Designation

Not all experiments carry equal weight in the paper. Clear labeling prevents scope creep and helps prioritize:

### Core (must appear in main body)

| Item | Section |
|---|---|
| Scenario A (same-domain) | Section 4.1 |
| Scenario B+C (few-shot transfer, LSTM + DLinear) | Section 4.2 |
| Scenario D.1 (stability evaluation) | Section 4.3 |
| Scenario D.2 (anomaly injection, 5%, key types) | Section 4.4 |
| Scenario D.3 (AD baseline comparison, LSTM-AE + IF) | Section 4.5 |
| Ablation E1, E2, E4 | Section 4.6 |
| Figure 1 (framework), Figure 2 (few-shot curve), Figure 3 (case study) | Sections 3, 4, 5 |

### Supplementary (appendix or separate file)

| Item |
|---|
| Full D.3 results with all 4 AD baselines |
| Ablation E3 (sqm), E5 (fine-tune strategy) |
| Scenario F (all sensitivity analyses) |
| PatchTST results (if completed) |
| GRU, TCN results |
| 2% anomaly injection results |
| All 7 anomaly types individually |
| Full statistical test tables |
| Per-building breakdown tables |

**Strategy**: Write the paper with core results in the main body. In the response letter, preempt typical reviewer requests ("we also tested X, see Appendix B") — this is a well-known Q2 success tactic.

---

## 12. Three Core Figures

### Figure 1: TL-TFAD Framework
- Type: System overview / flowchart
- Content: 3 modules (A → B → C) with data flow arrows
- Tool: draw.io or Python matplotlib + patches

### Figure 2: Few-Shot Performance Curve
- Type: Multi-panel line plot
- X-axis: $k$ (days), log scale
- Y-axis: MAE (left), $\sigma_{err}$ (right), F1@D.2 (right)
- Lines: M1, M2, M3, M5 (dashed)
- Facets: LSTM row, DLinear row
- Error bars: ±1 SE
- Key annotation: arrow pointing to where M3 crosses below M2

### Figure 3: Case Study
- Type: Time-series overlay
- Content: One week of target building test data
- Subplots stacked vertically:
  1. $y_{\text{true}}$ vs $\hat{y}$ (LSTM M2 vs M3)
  2. $|e_t|$ (prediction error)
  3. $s_t$ (anomaly score) with threshold line
  4. Detected anomaly markers
- Injected anomalies highlighted in red; true positives vs false positives marked differently

---

## 13. Risks and Mitigation

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| 1 | BDG2 inaccessible | L | H | Pre-download; have ASHRAE GEP III as backup. Document the switch in the paper. |
| 2 | < 6 buildings pass quality criteria | M | H | Relax to ≥ 6 months data per building; document this as a study limitation. |
| 3 | Transfer shows no benefit (all negative transfer) | L | H | **This is publishable.** Write the paper as "When and Why Transfer Learning Fails for Building Energy Forecasting." The experiments are identical; only the framing changes. |
| 4 | D.3 baselines (LSTM-AE) are complex to implement | M | M | Prioritize Isolation Forest + LOF (trivial). Implement LSTM-AE as a "deep unsupervised baseline." Skip OmniAnomaly if out of time. |
| 5 | 5 seeds × 6 buildings × 5 k × 3 methods × 2 models = combinatorial explosion | M | M | Cache pretrained models. Use a job scheduler (e.g., Slurm, or simple bash queue). Run the smallest $k$ first — if results are nonsense at $k=1$, fix before running all. |
| 6 | Synthetic anomalies don't convince reviewers | M | M | Frame D.2 as *supplementary validation*, not the primary claim. The primary claim is D.1 (stability on real data). Add a sentence: "We acknowledge that synthetic anomalies approximate real faults; future work should validate on buildings with operator-labeled anomaly logs." |
| 7 | Reviewer: "This isn't anomaly detection, it's just forecasting." | M | H | D.3 is the shield. If you don't compare with at least 2 actual anomaly detection methods, this criticism is valid. LSTM-AE + Isolation Forest comparison is non-negotiable for Q2. |

---

## 14. Self-Consistency Audit

| Check | Status |
|---|---|
| Each method module → experiment scenario | ✅ A→A, B→B+C, C→D |
| Each research question → specific analysis | ✅ |
| Core contribution (stability) → dedicated metrics in every table | ✅ $\sigma_{err}$, $\sigma_{score}$ in all output specs |
| "Anomaly detection" claim → AD baseline comparison | ✅ Scenario D.3 added |
| Q2 statistical rigor → tests + effect sizes + CI | ✅ Wilcoxon, Cohen's d, 95% CI |
| Negative/null results handled → framed as contribution | ✅ Scenario F.4 |
| Modern baselines present → DLinear + PatchTST | ✅ |
| Classical baselines present → ARIMA, SARIMA | ✅ |
| Reproducibility → seeds, config format, split definitions | ✅ |
| Study limitations acknowledged → D.2 caveat, building count | ✅ |

---

## 15. Final Assessment

**Q: Can this experimental design support an SCI Q2 paper?**

**Answer: Yes — if executed to this specification.**

The v2.0 design addresses all major gaps identified in v1.0:

| Gap in v1.0 | Resolution in v2.0 |
|---|---|
| No anomaly detection baseline | D.3: Isolation Forest, LOF, LSTM-AE |
| 3 buildings insufficient | 6–8 buildings, 3–4 types |
| No statistical tests | Wilcoxon, Diebold-Mariano, Cohen's d, 95% CI |
| No hyperparameter sensitivity | F.1: 5 parameter sweeps |
| Only 1 modern baseline | DLinear + PatchTST |
| No classical baselines | ARIMA, SARIMA, Naive |
| Anomaly injection generic | 7 building-realistic anomaly types |
| No negative transfer analysis | F.4: boundary condition analysis |
| No pretraining scale analysis | F.2: how many source buildings needed? |

**What still separates this from Q1 / top-tier (AAAI, NeurIPS, Applied Energy)?**

| Gap | Honest assessment |
|---|---|
| Theoretical analysis (generalization bounds, regret bounds) | Not needed for applied Q2 journal; would be needed for ML conference |
| Real anomaly labels | Ideal but infeasible for BDG2; synthetic injection with honest caveats is acceptable |
| Much larger scale (50+ buildings) | 6–8 is adequate for Q2 applied journal; 50+ would need a different dataset |
| Method novelty (new architecture/loss) | Not needed — New Setting papers are judged on problem framing + empirical rigor, not architectural novelty |

**Recommended target journal strategy (3-tier plan)**:

- **Tier 1 (stretch)**: Energy and Buildings (IF 6.7) or Building and Environment (IF 7.1) — needs all core experiments + PatchTST working + solid writing.
- **Tier 2 (realistic)**: Journal of Building Engineering (IF 6.4) or Energy (IF 9.0, short communication format) — this design is well-matched to their expectations.
- **Tier 3 (safety)**: Buildings (IF 3.0+) or Applied Sciences (IF 2.5+) — guaranteed acceptance if experiments are executed to this specification.

**The single most important success factor**: $\sigma_{err}$ and $\sigma_{score}$ MUST show clear separation between M3 (pretrain+FT) and M2 (target-only). If they don't, the paper's central claim collapses. Run the k=7 experiment FIRST — if the stability gap isn't there, diagnose before running the full matrix.

---

## 16. Next Implementation Steps

1. **Download BDG2** and verify data integrity.
2. **Scan buildings** against selection criteria; produce `table_dataset_statistics.csv`.
3. **Select 6–8 buildings** across 3–4 types.
4. **Implement preprocessing** → `data/processed/bdg2_electricity_hourly.csv`.
5. **Implement LSTM** training loop for Scenario A.
6. **Run k=7 experiment FIRST** (LSTM, 1 building, M2 vs M3) — validate the core claim before committing to the full matrix.
7. If $\sigma_{err}$ gap exists: proceed with full experiment plan.
8. If not: diagnose (wrong pretraining? architecture? building selection?) and iterate.
