# Mentor Requirements Compliance Audit

Date: 2026-05-19

This file checks the PA-MSR manuscript against the stricter research requirements repeatedly emphasized during project development: do not force a story, do not hide weak results, use strong baselines, make the method claim match the evidence, and keep the experiment protocol defensible.

## 1. Start from the real problem, not from module assembly

Status: mostly satisfied.

Evidence in manuscript:

- The Introduction frames the problem as cold-start next-hour building load forecasting with few target days.
- The motivation is persistence dominance, source-target mismatch, inactive/near-zero buildings, and negative transfer.
- The method is introduced as residual prediction around persistence, not as a generic module replacement.

Remaining risk:

- The method name PA-MSR is defensible, but the paper must not mention "module stitching" or "top-conference module replacement" anywhere in the submitted text.

Action:

- Keep DSOF/PA-DSOF in `docs/dsof_baseline_adaptation_plan.md` as an archived route only.

## 2. Use strong baselines, not only weak neural baselines

Status: satisfied for the current claim.

Required baselines:

- Persistence
- Seasonal naive
- RF-ST
- ExtraTrees-ST
- HistGBDT-ST
- Earlier residual boosting baseline
- DLinear neural transfer baselines where available
- Residual RevIN / module ablations

Evidence in manuscript:

- Table 2 compares PA-MSR/PA-MSR+ with persistence and the strongest tree baseline on BDG2-24.
- Table 4 reports PA-MSR scale validation on BDG2-120.
- Table 5 compares PA-MSR/PA-MSR+ with ExtraTrees-ST and persistence on COFACTOR-44.
- Table 6 reports paired robustness and negative-transfer rates.

Remaining risk:

- If the final journal version only keeps compact tables, reviewers may ask for the full leaderboard. The full tables must be included as supplementary tables or repository source data.

## 3. Do not overclaim method superiority

Status: mostly satisfied.

Claims currently allowed:

- PA-MSR is the default method for extreme cold start.
- PA-MSR+ is a conditional enhancement for longer adaptation windows.
- Persistence anchoring, residual RevIN, and MSR are the supported core mechanisms.
- COFACTOR-44 supports external active-building validation, not universal superiority.

Claims explicitly excluded:

- PA-MSR+ is not a universal full model.
- The final PA-MSR matrix was not exhaustively evaluated on all eligible BDG2 buildings.
- Forecasting MAE does not solve anomaly detection.
- Ordinary similarity weighting and gates are not supported contributions.

Action already taken:

- The manuscript includes a "what is not claimed" paragraph in Discussion.
- The COFACTOR section title was changed from "confirms the mechanism" to "supports external validation".

## 4. Explain why not the entire BDG2 dataset

Status: improved, but still a reviewer-risk point.

Current defense:

- BDG2-24 is used for the full method matrix and expensive neural comparisons.
- BDG2-120 active subset now directly validates PA-MSR against persistence, source-target tree ensembles, and the earlier residual boosting baseline.
- COFACTOR-44 is used as external active-building validation.
- Limitations explicitly state that the full neural matrix and all-eligible-BDG2 evaluation were not repeated at this scale.

Remaining risk:

- Energy and Buildings reviewers may still ask for a full neural matrix or an all-eligible-BDG2 run.

Best answer after the PA-MSR-only BDG2-120 run:

- Present BDG2-24 as a controlled full-matrix benchmark, BDG2-120 as PA-MSR scale robustness on active buildings, and COFACTOR-44 as external validation. Do not claim that the full neural matrix or all eligible BDG2 buildings were evaluated.

## 5. Handle inactive/near-zero buildings honestly

Status: satisfied.

Evidence:

- Table 3 reports the active-target subset.
- Figure 4 separates active and inactive/near-zero targets.
- The Results section warns that BDG2 median MAE must be interpreted carefully.

Remaining risk:

- The main text should avoid using median MAE as the main positive evidence.

## 6. Avoid leakage

Status: satisfied in writing, but code/protocol should remain auditable.

Evidence:

- Problem formulation states that the test period is never used for source selection, similarity computation, residual scaling, calibration, fitting, or hyperparameter selection.
- Experimental protocol states that validation is used only for model selection, early stopping, and calibrated residual amplitude.
- Figure 1 visualizes no-test-leakage data flow.

Remaining risk:

- PA-MSR+ uses validation calibration. This is acceptable only because the validation period is separated from test and not counted as target few-shot adaptation.

## 7. Make contribution look like a real paper

Status: substantially improved.

Current contribution package:

- Evaluation contribution: strict cold-start benchmark with strong baselines.
- Diagnostic contribution: direct neural transfer can fail badly under building heterogeneity.
- Method contribution: PA-MSR residual formulation with residual RevIN and MSR.
- Conditional extension: PA-MSR+ for moderate cold-start settings.
- Evidence contribution: paired, active/inactive, module, and external validation analyses.

Remaining risk:

- Method novelty is moderate, not breakthrough-level. The paper should be positioned as an energy/building applied-ML contribution, not a pure ML model paper.

## 8. Current recommendation

The manuscript is now aligned with the mentor's strict requirements at the framing and evidence-chain level after adding PA-MSR-only validation on the BDG2-120 active subset. The main remaining scale weakness is narrower: the full neural method matrix and all-eligible-BDG2 evaluation were not repeated.
