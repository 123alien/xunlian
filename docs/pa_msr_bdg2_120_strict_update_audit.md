# PA-MSR BDG2-120 Strict Update Audit

Date: 2026-05-20

Purpose: audit whether the manuscript and submission assets have been synchronized after the PA-MSR-only BDG2-120 lightweight validation finished.

## Verdict

Updated and currently consistent. The experiment result itself is strong and directly useful. The manuscript, figure plan, reviewer-risk audit, and submission checklist have been synchronized so that BDG2-120 is now described as PA-MSR-only scale validation rather than only earlier residual-boosting evidence.

## Evidence Now Available

Source files:

- `results/pa_msr_bdg2_120_light_cached_summary/table_pa_msr_bdg2_120_leaderboard.csv`
- `results/pa_msr_bdg2_120_light_cached_summary/table_pa_msr_bdg2_120_pairwise.csv`
- `results/pa_msr_bdg2_120_light_cached_summary/table_pa_msr_bdg2_120_building_level.csv`

PA-MSR leaderboard on BDG2-120:

| k | mean MAE | median MAE | P90 MAE | P95 MAE | worst MAE | mean rank | top-3 rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 16.52 | 10.47 | 40.71 | 47.01 | 95.65 | 1.308 | 98.3% |
| 7 | 16.43 | 10.82 | 40.31 | 45.70 | 92.87 | 1.467 | 95.0% |
| 14 | 16.20 | 10.80 | 39.61 | 44.00 | 88.33 | 1.375 | 98.3% |
| 30 | 15.84 | 10.13 | 37.04 | 43.09 | 87.29 | 1.408 | 95.8% |

Main paired evidence:

| Baseline | k=3 win / neg. transfer | k=7 win / neg. transfer | k=14 win / neg. transfer | k=30 win / neg. transfer |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 95.8% / 4.2% | 96.7% / 3.3% | 97.5% / 2.5% | 97.5% / 2.5% |
| RF-ST | 94.2% / 5.8% | 92.5% / 7.5% | 95.0% / 5.0% | 92.5% / 7.5% |
| ExtraTrees-ST | 95.8% / 4.2% | 91.7% / 8.3% | 91.7% / 8.3% | 92.5% / 7.5% |
| Prior residual boosting | 89.2% / 10.8% | 80.8% / 19.2% | 85.0% / 15.0% | 85.8% / 14.2% |

All above paired tests are significant by Wilcoxon signed-rank test over buildings.

## Step-by-Step Audit

### 1. Add BDG2-120 PA-MSR result to the manuscript

Status: fixed in `docs/manuscript_draft_parboost.md`.

Current problem:

- `docs/manuscript_draft_parboost.md` still says the 120-building subset is only evidence for the earlier persistence-anchored residual boosting baseline.
- This appears in the abstract/introduction context, dataset protocol, results Section 6.3, limitations, and supplementary-material notes.

Applied fix:

- Update the abstract to mention that PA-MSR was additionally validated on a 120-building active BDG2 robustness subset.
- Rename Section 6.3 from principle-only evidence to direct PA-MSR scale validation.
- Replace old PARBoost-only values with PA-MSR leaderboard and paired values.

Safe claim:

> PA-MSR remained the top-ranked method on a 120-building active BDG2 robustness subset, with mean ranks of 1.31-1.47 and top-3 rates above 95% across all target-data budgets.

Do not claim:

> PA-MSR was exhaustively evaluated on every eligible BDG2 building.

### 2. Update tables and supplementary tables

Status: fixed in the manuscript and available as source CSV/Markdown files.

Available:

- The new leaderboard and pairwise CSV/Markdown files exist locally.

Missing:

- The manuscript still has no BDG2-120 PA-MSR table.
- Supplementary Table S1 still describes only persistence, seasonal naive, RF-ST, ExtraTrees-ST, HistGBDT-ST, and earlier residual boosting.

Applied fix:

- Add a BDG2-120 robustness table to the main text or Supplementary Table S1.
- Include PA-MSR, prior residual boosting, persistence, RF-ST, ExtraTrees-ST, and HistGBDT-ST.
- Include mean MAE, median MAE, P90/P95, worst MAE, mean rank, and top-3 rate.

### 3. Update paired evidence

Status: fixed in Table 6 and retained in the local pairwise summary files.

Available:

- PA-MSR vs Persistence, RF-ST, ExtraTrees-ST, HistGBDT-ST, Seasonal naive, and prior residual boosting are available at building level for n=120.

Applied fix:

- Add BDG2-120 rows to Table 5 or create a new robustness paired table.
- Most important rows are PA-MSR vs Persistence, RF-ST, ExtraTrees-ST, and prior residual boosting.

Strict interpretation:

- PA-MSR clearly improves over classical baselines and prior residual boosting in paired comparisons.
- The largest remaining negative-transfer rate is against prior residual boosting at k=7: 19.2%. This should not be hidden; it supports the statement that PA-MSR improves but does not eliminate all building-level failures.

### 4. Re-adjust claims

Status: fixed.

Old claim still present:

> BDG2-120 supports the persistence-anchored principle.

New supported claim:

> BDG2-120 directly supports the scalability of PA-MSR on a larger active-building subset.

Boundary:

- BDG2-120 is a PA-MSR-only scale validation plus existing classical/prior residual baselines.
- It is not a full neural method matrix and not a full 1578-building BDG2 evaluation.

### 5. Update figures

Status: fixed.

Current issue:

- `scripts/make_pa_msr_paper_figures.py` currently generates Figure 1-4 from `results/final_claim_evidence`.
- It does not read `results/pa_msr_bdg2_120_light_cached_summary`.
- Existing PA-MSR figures were generated before the BDG2-120 PA-MSR result.

Applied fix:

- Either add a new main Figure 5 / Supplementary Figure S1 panel for BDG2-120 PA-MSR leaderboard, or expand Figure 2 to include a BDG2-120 panel.
- Recommended safest route: keep Figure 2 as BDG2-24 + COFACTOR-44, add Supplementary Figure S1 for BDG2-120 scale validation.

Minimum figure:

- x-axis: k = 3, 7, 14, 30.
- y-axis: mean MAE or mean rank.
- Methods: PA-MSR, prior residual boosting, Persistence, RF-ST, ExtraTrees-ST.

### 6. Mentor/journal-readiness audit

Status: fixed.

Files with outdated risk statements:

- `docs/mentor_requirements_compliance_audit.md`
- `docs/energy_buildings_submission_checklist.md`
- `docs/pa_msr_reviewer_risk_audit.md`

Applied fix:

- Replace "larger PA-MSR validation is missing" with "PA-MSR-only BDG2-120 validation completed; full neural matrix and full eligible-BDG2 evaluation remain unrun."
- Keep the limitation, but narrow it accurately.

## Reviewer Risk After This Update

Reduced:

- The objection "why did you not run PA-MSR on the larger BDG2 active subset?" is now largely addressed.

Still present:

- BDG2-120 is not the full neural matrix.
- BDG2-120 is an active subset, not all BDG2 buildings.
- PA-MSR+ was not run on BDG2-120 in this lightweight validation.
- Absolute MAE values should not be compared across BDG2-24, BDG2-120, and COFACTOR as if the target distributions are identical.

## Go/No-Go

Go for manuscript polishing and final consistency checks. Do not add new experiments before converting the synchronized manuscript, tables, and figures into the target journal format.
