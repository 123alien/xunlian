# Reviewer Risk Check For Forecasting Manuscript

## Current Manuscript Main Claim

Cross-building transfer improves few-shot neural building electricity forecasting, and DLinear-SRFT further improves standard DLinear transfer, becoming competitive with strong source+target tree baselines at k=7 and k=14.

## Major Risks

### Risk 1: The proposed method is not best at k=3

Reviewer concern:

The method is advertised as cold-start, but persistence and RandomForest source+target are stronger at k=3.

Defense:

- Do not claim universal superiority.
- State that k=3 is an ultra-cold-start limitation.
- Emphasize that DLinear-SRFT is strongest at k=7/k=14.
- Present k=3 as evidence that strong simple baselines are necessary.

Required manuscript wording:

> In the most extreme 3-day setting, persistence and source+target tree baselines remained stronger on average, indicating that ultra-cold-start forecasting may be dominated by short-term autocorrelation and requires different adaptation assumptions.

### Risk 2: DLinear-SRFT does not improve LSTM

Reviewer concern:

If source replay is useful, why does it hurt LSTM?

Defense:

- Frame DLinear-SRFT as an architecture-specific method.
- Do not call source replay a universal neural transfer solution.
- Explain that DLinear's lower-capacity structure may benefit more from source regularization.

Required manuscript wording:

> The replay mechanism was architecture-dependent: it consistently improved DLinear but not LSTM. Therefore, the proposed method is DLinear-SRFT rather than a generic replay rule for all neural forecasters.

### Risk 3: Classical baselines are strong

Reviewer concern:

Why use neural transfer if RandomForest source+target is similar or better?

Defense:

- The paper's value is not "neural always wins"; it is a careful cold-start transfer evaluation.
- DLinear-SRFT has competitive rank at k=7/k=14 and consistent improvement over standard neural transfer.
- Strong baselines improve credibility.

Required manuscript wording:

> The comparison with source+target tree models shows that neural transfer should not be evaluated only against target-only neural baselines.

### Risk 4: Dataset coverage

Reviewer concern:

12 buildings may be insufficient for broad BDG2 claims.

Defense:

- Avoid claiming full BDG2 generality.
- Say "12-building evaluation" clearly.
- Put broader BDG2-scale evaluation in limitations.

Possible repair:

- Add a table listing the 12 target buildings and building categories.
- Add subgroup analysis if building-type metadata are reliable.

### Risk 5: Statistical testing

Reviewer concern:

Seeds are not independent.

Defense:

- Main text uses building-level tests after seed averaging.
- Seed-level tables are supplementary.

Required manuscript wording:

> Repeated seeds were averaged within each target building before paired statistical testing, so the main inferential unit is the target building rather than the random seed.

### Risk 6: Method selection after seeing results

Reviewer concern:

Why select lambda=0.2?

Defense:

- Report all lambda values in supplementary.
- State lambda=0.2 was selected from a small predefined grid.
- Avoid pretending it was theoretically guaranteed.

Possible repair:

- Add supplementary table for lambda=0.05/0.1/0.2.
- If time allows, run one validation-based lambda selection protocol instead of global best. This would be stronger but not strictly necessary if the grid is transparently reported.

## Moderate Risks

### Need real references

The draft currently contains citation placeholders. Before manuscript submission, fill these with real citations for:

- BDG2 dataset.
- DLinear.
- building energy forecasting review or benchmark.
- building energy transfer learning.
- source replay or continual learning.

### Need stronger implementation details

Add:

- exact feature columns,
- window size and horizon,
- train/val/test split,
- model hyperparameters,
- optimizer, learning rates, epochs, early stopping,
- tree model hyperparameters,
- source replay lambda grid.

### Need table/figure numbering

Map final package outputs:

- Table 1: selected building summary.
- Table 2: M3 vs M2 building-level transfer statistics.
- Table 3: final method leaderboard.
- Figure 1: `figure_final_method_mae.png`.
- Figure 2: `figure_final_dlinear_transfer_curve.png`.
- Figure 3: `figure_final_m3r_delta_k*.png`.

## Current Readiness

Manuscript readiness: promising but not submission-ready.

Main missing pieces:

- real citations,
- dataset/building metadata table,
- polished Methods details,
- final figure/table integration,
- optional validation-based lambda selection explanation.

