# Experiment Machine Migration Plan

Date: 2026-05-25

This document is the migration checklist for moving the project from the old experiment machine
(`~/桌面/xunlian`) to a new experiment machine before the old machine is reinstalled.

## 1. Priority

Do this before any new experiments:

1. Create a remote backup archive on the old experiment machine.
2. Copy the archive and checksum back to the local workstation.
3. Verify the local archive checksum.
4. Copy the archive to the new experiment machine.
5. Restore and run a small smoke test.

## 2. What Must Be Preserved

Critical manuscript and evidence files:

- `docs/manuscript_draft_parboost.md`
- `docs/parboost_references.bib`
- `docs/parboost_references.ris`
- `docs/pa_msr_numeric_lock_audit.md`
- `docs/pa_msr_bdg2_120_strict_update_audit.md`
- `docs/mentor_requirements_compliance_audit.md`
- `figures/pa_msr/`
- `results/final_claim_evidence/`

Critical experiment results:

- `results/pa_msr_bdg2_120_light_cached_summary/`
- `results/pa_module_screen_cofactor_44_msr_dps/`
- `results/pa_module_extensions_cofactor_44/`
- `results/bdg2_robustness_120/`
- `results/pa_srft_pilot_bdg2/`
- `results/cofactor_external/`
- `results/phase2_paper/expanded_fast/`

Critical data and code:

- `scripts/`
- `src/`
- `configs/`
- `data/processed/`
- `requirements.txt`
- root-level `run_*.py` scripts

Optional but useful:

- `external/`
- `data/raw/` if present and not too large
- older `results/phase1/`, `results/phase2/`, and pilot folders if disk space allows

## 3. Scripts

The migration scripts are:

- `scripts/remote_make_migration_backup.sh`
- `scripts/local_fetch_migration_backup.ps1`
- `scripts/new_machine_restore_backup.sh`

Run order:

```bash
# On local Windows PowerShell, after SSH works:
ssh lrh@100.118.41.2 'cd ~/桌面/xunlian && bash scripts/remote_make_migration_backup.sh'
```

```powershell
# On local Windows PowerShell:
powershell -ExecutionPolicy Bypass -File scripts\local_fetch_migration_backup.ps1 `
  -Remote lrh@100.118.41.2 `
  -RemoteProject '~/桌面/xunlian' `
  -LocalBackupDir 'C:\训练\machine_backups'
```

```bash
# On the new experiment machine, after uploading the archive:
bash new_machine_restore_backup.sh ~/xunlian_migration_YYYYMMDD_HHMMSS.tar.gz ~/桌面/xunlian
```

## 4. Verification After Restore

After restore, run:

```bash
cd ~/桌面/xunlian
python --version
python -c "import pandas, numpy, sklearn; print('basic imports ok')"
python scripts/make_pa_msr_paper_figures.py
```

The figure script should regenerate:

- `figures/pa_msr/fig1_protocol_no_leakage.*`
- `figures/pa_msr/fig2_main_leaderboard.*`
- `figures/pa_msr/fig3_pairwise_robustness.*`
- `figures/pa_msr/fig4_bdg2_active_inactive.*`
- `figures/pa_msr/figS1_bdg2_120_pa_msr_scale_validation.*`

## 5. Current Known Issue

As of 2026-05-25, SSH to `lrh@100.118.41.2` timed out from the local workstation. Before backup, confirm that:

- the old machine is powered on;
- Tailscale or the private network is connected;
- SSH service is running;
- the address `100.118.41.2` is still correct.

