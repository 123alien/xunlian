# Cold-Start Transfer Forecasting Research Protocol

## Research Question

Can cross-building transfer learning improve few-shot hourly building energy
forecasting for newly observed active buildings?

The target scenario is not inactive-state detection and not long-history
forecasting. It is active-load forecasting when the target building has only a
small adaptation window.

## Primary Scientific Claims To Test

1. Target-only neural models are unstable under few-shot target data.
2. Cross-building pretraining improves neural adaptation under the same target
   data budget.
3. Source replay fine-tuning improves DLinear-style transfer over standard
   target-only fine-tuning.
4. Classical and naive baselines remain important reference points; the neural
   method must be evaluated against them rather than only against neural
   target-only baselines.

## Dataset Inclusion Criteria

Datasets are eligible if they provide:

- multiple buildings;
- hourly or sub-hourly electricity or total energy time series;
- enough history to support chronological train/validation/test splits;
- enough buildings to form source-target transfer tasks;
- target test periods that represent active load forecasting.

## Building Inclusion Criteria

A building is eligible for the primary active-load benchmark if, using only the
target time series:

- test zero ratio is below 0.5;
- test mean load is greater than a small positive threshold;
- test standard deviation is greater than a small positive threshold;
- persistence MAE is greater than a small positive threshold;
- the building has enough data to construct 24-hour input windows.

Buildings failing these criteria are not deleted silently. They are reported as
inactive or near-constant operating-state-shift cases.

## Data Split

For each building:

- first 60% chronological data: source pretraining or target few-shot pool;
- next 20%: validation;
- final 20%: test.

The target few-shot pool uses the first k days from the target training segment.
Validation and test periods are never used for target adaptation.

## Few-Shot Budgets

Primary:

- k = 3, 7, 14, 30 days.

If a dataset has strong weekly seasonality, k = 7 and k = 14 are emphasized in
the main text, while k = 3 is treated as an extreme cold-start setting.

## Required Baselines

Every dataset must include:

- persistence;
- seasonal naive, when the timestamp granularity supports it;
- Random Forest source-target;
- ExtraTrees source-target or another robust tree ensemble;
- DLinear target-only;
- DLinear pretrain plus fine-tune;
- DLinear source replay fine-tune.

Optional:

- LSTM transfer;
- PatchTST or another modern sequence model, when the task length and data
  volume justify it.

## Metrics

Primary:

- MAE;
- normalized MAE relative to building mean load;
- mean rank across buildings;
- building-level win rate.

Secondary:

- RMSE;
- sMAPE, with near-zero caveats;
- sigma_err for residual stability.

Seeds are aggregated within each building-k-method unit before scientific
comparisons. Buildings, not seeds, are the independent units.

## Reporting Structure

1. Full benchmark: all eligible raw buildings, including inactive/near-constant
   cases.
2. Active-load benchmark: primary forecasting benchmark.
3. Operating-state-shift analysis: inactive or near-constant buildings.
4. External validation: repeat the same protocol on additional datasets.

## Go/No-Go Criteria For A Method-Centered Paper

DLinear-SRFT can be a central method claim only if it:

- consistently improves DLinear pretrain plus fine-tune across active buildings;
- reduces worst-case failures compared with direct DLinear fine-tuning;
- is competitive with tree source-target baselines on at least one external
  active-load dataset or under clearly defined active-load conditions.

If it only improves neural target-only baselines but remains clearly below tree
methods, the paper should be framed as an empirical transfer-learning benchmark
rather than a dominant-method paper.
