# Next Experiment Queue After Forecasting Reposition

## Current Finding After Adding Classical Baselines

The forecasting story is stronger but more nuanced:

- M3 strongly outperforms M2 for both LSTM and DLinear.
- Simple and classical source+target baselines are very competitive.
- `random_forest_source_target`, `extra_trees_source_target`, and `ridge_source_target` beat or match neural M3 in average MAE in several k settings.
- Therefore, the paper should not claim that neural transfer is the best overall forecasting method yet.

## Immediate Interpretation

The current evidence supports:

> Neural pretraining plus fine-tuning is highly effective compared with target-only neural training under cold-start conditions.

The current evidence does not yet support:

> Neural transfer learning is the best method among all forecasting baselines.

## Next Experiments

### Experiment 1: Replay Fine-Tuning For Neural Transfer

Problem:

M3 pretrains on source buildings, then fine-tunes only on a few target windows. This may forget useful source structure and overfit the tiny target set.

New method:

- M3R: source pretraining + target fine-tuning with source replay.
- During fine-tuning, each batch mixes:
  - target few-shot windows,
  - a small replay sample from source buildings.
- Use target-weighted loss:
  - `loss = target_loss + lambda_source * source_loss`
  - try `lambda_source = 0.05, 0.1, 0.2`.

Expected benefit:

- Better generalization than pure target fine-tuning.
- More competitive against source+target tree models.

### Experiment 2: Frozen-Backbone Fine-Tuning

Problem:

Fine-tuning all weights on only 3 days of target data may over-adapt.

New methods:

- M3F-head: freeze recurrent/decomposition backbone and train only final projection.
- M3F-last: freeze early layers and fine-tune only later layers.

Expected benefit:

- Better k=3 stability.
- Lower negative transfer risk.

### Experiment 3: Pooled Neural Baseline

Problem:

Classical source+target baselines directly train on pooled source and target data. Neural M3 uses pretraining then target-only fine-tuning, which is not exactly the same adaptation mode.

New method:

- M4 source+target pooled neural training.
- Train the same LSTM/DLinear on source windows plus target k-day windows from scratch or from pretrained weights.

Expected benefit:

- Fairer comparison with source+target RandomForest and ExtraTrees.
- Helps determine whether the issue is the neural architecture or the fine-tuning protocol.

### Experiment 4: Model Capacity And Regularization Ablation

Problem:

The neural models may be too large for few-shot target adaptation.

Run:

- LSTM hidden_dim = 32, 64, 128.
- dropout = 0.1, 0.2, 0.4.
- weight_decay = 1e-4, 1e-3.

Expected benefit:

- Smaller or more regularized LSTM may close the gap to tree baselines.

## Go/No-Go Criteria

Use the paper as a stronger forecasting paper if one of these holds:

- M3R or M4 becomes competitive with `random_forest_source_target` in k=3/k=7.
- M3R improves over M3 in building-level MAE and sigma_err without increasing negative transfer.
- At minimum, neural transfer remains clearly better than target-only neural training and the paper honestly reports tree baselines as strong competitors.

## Recommended Run Order

1. M3R replay fine-tuning with lambda_source = 0.1.
2. M4 pooled neural training.
3. Frozen-head fine-tuning for LSTM.
4. Regularization ablation only if the first three do not improve enough.

