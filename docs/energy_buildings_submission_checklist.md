# Energy and Buildings / Journal of Building Engineering Submission Checklist

Date: 2026-05-19

This checklist tracks journal-readiness for the PA-MSR manuscript. It follows the common Elsevier requirements for Energy and Buildings / Journal of Building Engineering style submissions.

## Completed in the manuscript

- Title page placeholder added.
- Abstract and keywords retained near the front of the manuscript.
- Highlights added as five short bullet points.
- Graphical abstract file identified: `figures/pa_msr/fig1_protocol_no_leakage.svg`.
- Main sections reorganized into journal-style structure:
  - Introduction
  - Related Work
  - Problem Formulation
  - Materials and Methods
  - Experimental Setup and Evaluation Protocol
  - Results
  - Discussion
  - Limitations
  - Conclusion
- The draft no longer contains a "Tables and Figures Plan" section.
- Figure captions added for Figures 1-4.
- Supplementary material section added.
- Funding, competing interest, data availability, code availability, CRediT, and AI-use declarations added.
- Main PA-MSR figures generated in SVG, PDF, PNG, and TIFF:
  - `figures/pa_msr/fig1_protocol_no_leakage.*`
  - `figures/pa_msr/fig2_main_leaderboard.*`
  - `figures/pa_msr/fig3_pairwise_robustness.*`
  - `figures/pa_msr/fig4_bdg2_active_inactive.*`
  - `figures/pa_msr/figS1_bdg2_120_pa_msr_scale_validation.*`
- Figure source data generated under `figures/pa_msr/source_data/`.

## Must be completed by the authors before submission

- Replace title-page placeholders with author names, affiliations, corresponding author email, and ORCID identifiers.
- Fill in funding information or confirm the no-specific-funding statement.
- Fill in the CRediT authorship contribution statement.
- Deposit code, processed manifests, evidence tables, and figure source data in a public repository, then replace repository placeholders in Data Availability and Code Availability.
- Confirm whether the target journal requires a separate graphical abstract upload and/or separate highlights file.
- Convert the manuscript from Markdown to the journal submission format, usually Word or LaTeX, while keeping editable figure text in SVG/PDF.

## Remaining scientific checks

- Keep the limitation that the BDG2-120 validation is PA-MSR-only and does not repeat the full neural matrix or all eligible BDG2 buildings.
- Do not describe PA-MSR+ as the final full model; it is a conditional enhancement.
- Do not claim anomaly detection performance from forecasting MAE.
- Do not import metric values from prior papers into the leaderboard unless the protocol is identical.

## Current highest-risk reviewer question

"Why is the full neural/PA-MSR matrix not evaluated on all BDG2 buildings?"

Current answer: BDG2-24 is used for the full method matrix and detailed neural comparison; BDG2-120 directly validates PA-MSR at larger active-building scale against persistence, source-target tree ensembles, and the earlier residual boosting baseline; COFACTOR-44 is used as external active-building validation. The remaining limitation is that BDG2-120 is not a full neural matrix and not an all-eligible-BDG2 evaluation.
