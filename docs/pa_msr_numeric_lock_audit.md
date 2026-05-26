# PA-MSR Numeric Lock Audit

Date: 2026-05-20

Purpose: verify that manuscript-facing numerical claims and tables in `docs/manuscript_draft_parboost.md` match the source CSV evidence tables.

## Source Tables Checked

- `results/final_claim_evidence/table_final_leaderboard_ranked.csv`
- `results/final_claim_evidence/table_bdg2_24_active_inactive_msr.csv`
- `results/final_claim_evidence/table_final_pairwise_robustness.csv`
- `results/pa_msr_bdg2_120_light_cached_summary/table_pa_msr_bdg2_120_leaderboard.csv`
- `results/pa_msr_bdg2_120_light_cached_summary/table_pa_msr_bdg2_120_pairwise.csv`

## Locked Manuscript Tables

### Table 2: BDG2-24 main cold-start benchmark

Status: locked.

Checked fields:

- best method by mean MAE for k = 3, 7, 14, 30
- mean MAE
- median MAE
- worst MAE
- strongest tree baseline mean MAE
- persistence mean MAE

Result: all values match the source table after manuscript rounding.

### Table 3: BDG2-24 active-target sensitivity

Status: locked.

Checked fields:

- active-target best residual variant for k = 3, 7, 14, 30
- mean MAE
- median MAE
- P90 MAE
- residual RevIN mean MAE

Result: all values match `table_bdg2_24_active_inactive_msr.csv` after manuscript rounding.

### Table 4: BDG2-120 active robustness validation

Status: locked.

Checked fields:

- PA-MSR mean MAE
- median MAE
- P90 MAE
- P95 MAE
- worst MAE
- mean rank
- top-3 rate

Result: all values match `table_pa_msr_bdg2_120_leaderboard.csv` after manuscript rounding.

Important boundary:

- PA-MSR is top-ranked by mean MAE and mean rank across all k values.
- PA-MSR is not claimed to be best on every robustness statistic; in particular, the worst-case MAE is not always the lowest among all methods.

### Table 5: COFACTOR-44 external active-building validation

Status: locked.

Checked fields:

- best method by mean MAE for k = 3, 7, 14, 30
- mean MAE
- median MAE
- worst MAE
- ExtraTrees-ST mean MAE
- persistence mean MAE

Result: all values match the source table after manuscript rounding.

### Table 6: Building-paired robustness evidence

Status: locked for the checked rows.

Checked rows:

- BDG2-120 vs Persistence, k = 3, 7, 14, 30
- BDG2-120 vs ExtraTrees-ST, k = 3, 7, 14, 30
- BDG2-120 vs prior residual boosting, k = 3, 7, 14, 30

Result: all values match `table_pa_msr_bdg2_120_pairwise.csv` after manuscript rounding.

## Narrative Claims Checked

Checked and consistent:

- Abstract: BDG2-120 top-ranked claim is now limited to mean MAE and mean rank.
- Methods: PA-MSR fixes `alpha_i = 1`; PA-MSR+ selects `alpha_i` on the validation period.
- COFACTOR: source dataset has 45 public buildings; benchmark retains 44 active buildings after coverage/activity filtering.
- BDG2-120: mean MAE decreases from 16.52 at k=3 to 15.84 at k=30.
- BDG2-120: mean rank ranges from 1.308 to 1.467.
- BDG2-120 vs Persistence: win rates and negative-transfer rates match the pairwise table.
- COFACTOR-44 vs ExtraTrees-ST: paired win rates 97.7%, 75.0%, 65.9%, and 88.6% match the source table.
- BDG2-24 vs residual RevIN: win-rate range 79.2-87.5% matches the source table.

## Automated Checks Run

Two local Python checks were run from PowerShell:

- Markdown Table 2, Table 4, Table 5, and BDG2-120 rows of Table 6 against source CSV values.
- Markdown Table 3 against `table_bdg2_24_active_inactive_msr.csv`.

Both returned:

```text
all_checked_ok
table3_ok
```

Figure generation was also rerun:

```text
python scripts\make_pa_msr_paper_figures.py
```

and completed successfully.

## Remaining Numeric Risks

- Figure source data should be checked visually against final manuscript captions after journal-format conversion.
- If any table is edited manually in Word/LaTeX, this numeric lock must be repeated.
- Supplementary tables not embedded in the manuscript should be exported directly from the locked CSV files rather than retyped.
