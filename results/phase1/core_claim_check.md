# Phase 1: Core Claim Validation Report

**Date**: 2026-05-10 18:19
**Seed**: 42
**k**: 7 days (168 hours)
**Buildings**: 5 (Bear_education_Bob, Robin_office_Adolph, Wolf_education_Tori, Wolf_public_Norma, Wolf_science_Alfreda)
**Model**: LSTM (hidden=128, layers=2)

## Core Claim

> Pretrain + fine-tune (M3) produces **lower σ_err and σ_score** 
> than target-only few-shot (M2), indicating more stable prediction 
> errors and more reliable anomaly scoring.

## Results

| Target Building | σ_err M2 | σ_err M3 | Δ σ_err | σ_score M2 | σ_score M3 | Δ σ_score | σ_err Win? | σ_score Win? |
|————————————————|——————————|——————————|—————————|————————————|————————————|———————————|———————————|—————————————|
| Bear_education_Bob | 51.6665 | 21.7714 | -29.8951 | 4.6594 | 2.1918 | -2.4676 | Win | Win |
| Robin_office_Adolph | 4.6090 | 10.1491 | +5.5401 | 12239518.5975 | 13.1876 | -12239505.4098 | Lose | Win |
| Wolf_education_Tori | 114.8005 | 110.1001 | -4.7004 | 4.2144 | 5.1618 | +0.9474 | Win | Lose |
| Wolf_public_Norma | 66.4188 | 33.5560 | -32.8628 | 8.2759 | 3.9933 | -4.2825 | Win | Win |
| Wolf_science_Alfreda | 11.0316 | 10.8018 | -0.2298 | 7.1481 | 6.1727 | -0.9754 | Win | Win |

## Summary

- **σ_err**: M3 wins in 4/5 buildings (mean Δ = -12.4296)
- **σ_score**: M3 wins in 4/5 buildings (mean Δ = -2447902.4376)

## Verdict

**SUPPORTED: M3 (pretrain+FT) produces more stable predictions than M2 (target-only) in the majority of buildings.**

### Interpretation

The core claim is supported. Transfer learning (pretrain + fine-tune) 
improves prediction stability compared to training on limited target data alone. 
This provides evidence that the TL-TFAD framework is viable, and the project 
should proceed to Phase 2 (full experiment matrix).

## Next Steps

If Phase 1 passes: proceed to `src/phase2.py` — full experiment matrix with 
k ∈ {0, 1, 3, 7, 14}, DLinear baseline, 5+ buildings, anomaly injection, 
and unsupervised AD baselines.