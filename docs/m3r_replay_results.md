# M3R Replay Fine-Tuning Results

## Main-Line Reminder

The paper main line is cold-start building energy forecasting. M3R was tested to improve neural transfer forecasting and to compete more fairly with strong source+target classical baselines.

## Method

M3R starts from the existing source-only pretrained checkpoint (`M1`, k=0), then fine-tunes on k-day target data with source replay:

`loss = target_loss + lambda_source * source_loss`

Tested lambda values:

- 0.05
- 0.1
- 0.2

## Result Summary

M3R is useful for DLinear but not for LSTM.

### DLinear

Compared with original DLinear M3, DLinear M3R improves MAE for all 12 buildings at every k setting.

Best lambda is usually 0.2:

- k=3: mean MAE delta vs M3 = -1.803, building win rate = 100%.
- k=7: mean MAE delta vs M3 = -1.389, building win rate = 100%.
- k=14: mean MAE delta vs M3 = -1.595, building win rate = 100%.

Residual stability also improves most clearly at k=7 and k=14:

- k=7, lambda=0.2: sigma_err delta = -0.262, building win rate = 91.7%.
- k=14, lambda=0.2: sigma_err delta = -0.346, building win rate = 100%.

### LSTM

LSTM M3R generally worsens MAE compared with original LSTM M3.

- k=3, lambda=0.05: mean MAE delta = +2.982.
- k=7, lambda=0.05: mean MAE delta = +0.750.
- k=14, lambda=0.05: mean MAE delta = +0.791.

Therefore, LSTM M3R should not be used as a main positive result.

## Comparison With Classical Baselines

DLinear M3R becomes competitive with source+target tree baselines in k=7 and k=14:

- k=7:
  - RandomForest source+target mean MAE = 5.913.
  - DLinear M3R lambda=0.2 mean MAE = 6.312.
  - DLinear M3R lambda=0.2 has strong sigma_err = 6.239.

- k=14:
  - RandomForest source+target mean MAE = 5.808.
  - DLinear M3R lambda=0.2 mean MAE = 5.926.
  - DLinear M3R lambda=0.2 has the best mean-rank profile among neural methods and is very close to RandomForest.

k=3 remains difficult:

- Persistence and source+target tree baselines are still much stronger than neural transfer methods on average.
- This should be discussed honestly as an ultra-cold-start limitation.

## Recommended Paper Claim

Use:

> Source replay fine-tuning further improves DLinear transfer forecasting, reducing MAE relative to standard pretrain-then-fine-tune adaptation for every target building and bringing the neural transfer model close to strong source+target tree baselines at k=7 and k=14.

Avoid:

> Source replay improves all neural transfer models.

Avoid:

> M3R is universally the best forecasting method.

## Main Paper Use

Recommended:

- Include DLinear M3R lambda=0.2 as the enhanced neural transfer variant.
- Put LSTM M3R in supplementary or limitations.
- Present RandomForest/ExtraTrees source+target as strong classical competitors, not weak baselines.

