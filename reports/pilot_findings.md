# Baseline pilot findings

Date: 2026-10-03
Method: `check_stability_data.py pilot` — positional peptide one-hot +
allele identity, Ridge(alpha=10), target log1p(hours). Peptide-grouped
~60/20/20 pool/dev/final-test partitions (seeds 2026/2027); random
acquisition order, 5 seeds; development-set evaluation only; final test
untouched.

## Inputs

- `data/processed/eligible_records.csv` — 5,815 batch-approved pairs
  (see `reports/source_review.md`), protocol `SPA_PMID21044632_37C`,
  protocol-derived temperature 37 °C.
- `data/processed/eligible_hla_a0201.csv` — 977-row A*02:01 subset
  (one pair per peptide).

## Results (development RMSE, log1p hours; mean over 5 seeds)

### Within-allele pilot (HLA-A*02:01) — `results/pilot_a0201/`

Pool 585 / dev 196 / final test 196 pairs.

| labels | model RMSE | const RMSE | model MAE |
| ---: | ---: | ---: | ---: |
| 29 | 1.066 ± 0.050 | 1.128 | 0.931 |
| 58 | 1.018 ± 0.027 | 1.109 | 0.887 |
| 146 | 0.963 ± 0.019 | 1.100 | 0.827 |
| 292 | 0.915 ± 0.018 | 1.099 | 0.772 |
| 585 | 0.846 | 1.098 | 0.702 |

### Pan-allele pilot (10 alleles) — `results/pilot_all_alleles/`

Pool 3,457 pairs (2,560 peptide groups) / dev 1,188 / final test 1,170.
Peptides shared across alleles are grouped into the same partition.

| labels | model RMSE | const RMSE | model MAE |
| ---: | ---: | ---: | ---: |
| 173 | 1.017 ± 0.012 | 1.071 | 0.833 |
| 346 | 0.992 ± 0.016 | 1.072 | 0.806 |
| 864 | 0.947 ± 0.005 | 1.072 | 0.768 |
| 1728 | 0.918 ± 0.007 | 1.071 | 0.742 |
| 3457 | 0.898 | 1.071 | 0.724 |

## Interpretation

- The labels carry learnable signal: a trivial sequence model beats
  the constant baseline at every budget in both pilots, with monotone
  learning curves. Label quality is sufficient to proceed.
- Curves have not plateaued at full pool — headroom remains, so
  retrospective label-efficiency differences between acquisition
  policies should be measurable.
- Per-pair prediction is noisy (best dev RMSE ≈0.85–0.90 log1p h,
  ≈ factor ~2.3 in hours). One-hot + Ridge is a weak baseline;
  embeddings may do better, but that is a hypothesis, not a result.

## Decision: PROCEED to active-learning comparison

Supported by: verified source chain (37 °C protocol-derived),
0 internal label conflicts, near-perfect agreement with the Rasmussen
snapshot, and positive pilot learning curves.

Constraints carried forward (from handoff §9–10):
- No SPEARMINT stability-trained checkpoint for embeddings —
  5,807/5,815 eligible pairs appear in its files. Use a general
  pretrained protein model (e.g., ESM-2) with recorded checkpoint.
- Same predictor for all three policies; peptide-grouped budgets
  counted in pair-level measurements; final test stays untouched until
  design lock.
- Engineered alleles / modified peptides are absent from this subset
  (all canonical, unmodified) — no special handling needed.

## Known limitations

- Positive-recorded-half-life population only (zeros excluded);
  binder-enriched pool; within-dataset retrospective result — not
  evidence of laboratory savings.
- Near-neighbour peptide leakage across partitions is possible.
- Dev-set results only; final-test evaluation is reserved for the
  locked AL design.
