# Active-learning comparison: results

Date: 2026-10-03
Dataset: 5,815 eligible SPA pairs (10 alleles), partitions from
`results/pilot_all_alleles/partitioned_data.csv`
(pool 3,457 / dev 1,188 / final test 1,170; peptide-grouped).
**Final test was never evaluated.**

## 1. Frozen embeddings

- Checkpoint `facebook/esm2_t33_650M_UR50D`, frozen, mean-pooled last
  hidden layer (BOS/EOS excluded). Pair feature =
  concat(peptide_emb, mhc_emb), 2×1280 dims.
- Generated once on Modal T4 via `scripts/modal_embed.py`: benchmark
  73 seq/s, 3.1 GB peak; full 4,268 peptides + 10 MHC chains in ~24 s
  (~$0.01). Cached at `data/processed/embeddings_esm2.npz` with
  `embeddings_meta.json`.
- MHC sequences: canonical full-length heavy chains from the SPEARMINT
  splits; A*02:01 uses the canonical variant (engineered K90A/E87Q
  excluded).
- A SPEARMINT stability-trained checkpoint was NOT used (5,807/5,815
  eligible pairs appear in its files — `reports/overlap_report.json`).

## 2. Predictor comparison (matched random acquisition)

Identical Ridge(alpha=10, lsqr) for both representations
(`results/predictor_comparison/`). Dev RMSE, log1p(hours):

| labels | one-hot | ESM-2 |
| ---: | ---: | ---: |
| 173 | 1.016 | 1.263 |
| 1038 | 0.936 | 1.189 |
| 2076 | 0.912 | 1.115 |

**One-hot beats ESM-2 at every budget under the identical fixed-alpha
procedure.** Diagnostic follow-up showed most of the gap is a
regularisation scaling artifact: on 2560-dim standardized embeddings,
alpha=10 underfits (alpha≈300–1000 optimal). With `RidgeCV` (alpha
chosen by internal LOO-CV on acquired labels only), ESM-2 reaches
~0.93 at 2,076 labels — still slightly behind one-hot (0.912), and
mean-pooling over 8–13-residue peptides plausibly destroys the
position-specific anchor information one-hot encodes directly.

## 3. Three-policy comparison (predefined design)

Locked in `results/policy_comparison/design.json` before running:
- Predictor identical across policies: `RidgeCV(alphas=1..3000)` on
  StandardScaler(pool-inputs) ESM-2 pair features.
- Matched initial set: same random 173 pairs per seed for all three
  policies. Batch 173; budgets 173→2,076 (5–60% of pool); 10 seeds.
- `seqdiv`: greedy k-center (max-min Euclidean) in one-hot sequence
  space; `embdiv`: same rule in standardized ESM-2 space; `random`.
  Scores recomputed vs. current labelled set and within-batch picks.
- Primary metric: dev RMSE(log1p h); scalar: AULC over log10(labels).

### Result — negative for both diversity policies

Dev RMSE at final budget (2,076): **random 0.937**, embdiv 0.957,
seqdiv 0.953. Random was best at **every** budget.

| policy | AULC (mean±sd) | Δ vs random | seeds better |
| --- | ---: | ---: | ---: |
| random | 1.073 ± 0.011 | — | — |
| embdiv | 1.086 ± 0.012 | +0.013 | 0/10 |
| seqdiv | 1.089 ± 0.007 | +0.016 | 0/10 |

Diversity acquisition *hurt* in all 10 seeds. Interpretation: greedy
max-min selection preferentially acquires outliers — unusual peptide–
allele combinations that are least like the labelled set — which are
low-information for a ridge predictor and may include atypical
measurements. Random acquisition samples the bulk distribution more
efficiently. Within this design, protein-embedding diversity does not
improve label efficiency over random; the honest answer to the project
question is **no** under this predictor/acquisition family.

## 4. Sequence-similarity sensitivity

`results/sensitivity/` — identity = best ungapped alignment of the
shorter peptide into the longer.

- Only 1.7% of dev peptides have a 100%-identity pool neighbour
  (cross-length substring cases; exact same-length duplicates were
  already grouped). ≥90%: 2.0%; ≥80%: 4.7%.
- Restricting dev RMSE at 2,076 labels to peptides with <80% or <70%
  identity to the acquired set leaves the ranking and gaps unchanged
  (random ≈0.937, diversity policies ≈0.95). The diversity-policy
  disadvantage is not an artifact of near-sequence leakage.

## 5. Caveats and limitations

- Retrospective label efficiency only; no laboratory-savings claim.
- Population: positive recorded half-lives from NetMHCpan-predicted
  binders, single SPA protocol family; zeros excluded.
- Mean-pooled ESM-2 + ridge is one (weak-ish) representation/predictor
  pair; a different predictor or non-mean pooling could change which
  policy wins — the negative result is specific to this design.
- Diversity-without-uncertainty is a weak AL family; uncertainty- or
  density-weighted policies remain untested.
- One-allele-per-submission structure means allele-level and
  study-level effects are confounded.
