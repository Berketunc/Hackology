# Source review: eligible peptide–HLA-I stability subset

Date: 2026-10-03
Scope: the 5,815 positive, unflagged IEDB records carrying the known
SPA replicate-count comment (`data/processed/eligible_records.csv`).

## Verified facts

1. Every eligible row's comment cites the SPA assay paper
   **PMID 21044632** (Harndahl et al., "Real-time, high-throughput
   measurements of peptide-MHC-I dissociation using a scintillation
   proximity assay", J Immunol Methods 2010).
2. All 10 IEDB submission references covering the subset
   (1028282, 1028285, 1028287–1028294; title "Large scale analysis of
   peptide - HLA-I stability") carry the **identical submission
   abstract**, fetched 2026-10-03:
   > "These data have been generated using a scintillation proximity
   > assay based peptide-HLA-I dissociation assay (PMID: 21044632).
   > HLA-I molecules were selected to cover known HLA supertypes
   > (using PMID: 14963618). Peptide ligands were selected by predicted
   > binding to the respective HLA-I molecule using the NetMHCpan
   > server (PMID: 19002680)"
3. PMID 21044632 §2.5 (verified from PMC4341823 full text,
   2026-10-03): complexes are refolded overnight at **18 °C**;
   dissociation is initiated by adding unlabeled β2m/peptide and
   **raising the temperature to 37 °C**, monitored on a TopCount
   "modified to run at 37 °C … placed in a temperature-controlled room
   set at 37 °C". The reported half-life is therefore a **37 °C
   dissociation** measurement; 18 °C is the association stage only.
4. Label cross-check against the Rasmussen sheet
   (`data/external/rasmussen_et_al_dataset.csv`): 5,517/5,815 eligible
   pairs present; 5,490 match exactly, remaining 27 differ by ≤0.0033 h
   (rounding). All 298 non-matching pairs are non-9-mers (lengths
   8, 10, 11, 13); the sheet contains only 9-mers.
5. Overlap with SPEARMINT files (`reports/overlap_report.json`):
   5,807/5,815 eligible pairs appear in at least one SPEARMINT file;
   4,420 in the stability-training split. **A SPEARMINT
   stability-trained checkpoint must not provide embeddings** for an
   experiment that treats these labels as hidden.

## Annotation decisions

- `approved=True` — batch approval at submission level, justified by
  the uniform submission abstract (per-reference evidence above), not
  by flag absence alone.
- `temperature_C=37.0` — **protocol-derived** inference level, not a
  per-row recorded field.
- `protocol_id=SPA_PMID21044632_37C`.
- `half_life_hours` = `half_life_hours_raw` (minutes/60); original
  values and units preserved.

## Unresolved assumptions (documented, not resolved)

- **Protocol adherence**: submissions state the SPA assay of
  PMID 21044632 was used; whether each run followed it unmodified
  (e.g., 37 °C dissociation, β2m-chase format) is assumed, not
  verified per submission.
- **Zeros excluded**: the 482 zero-valued records in the same
  submissions likely encode "below detection/unstable"; pilot
  population is *positive recorded half-lives only* — do not
  generalise to nonbinders.
- **Replicate averaging**: labels are averages of ≥2 experiments;
  per-replicate variance is unavailable.
- **Peptide selection bias**: ligands were NetMHCpan-predicted
  binders; the pool is enriched for binders, not a random peptide
  sample.
- **Near-sequence leakage**: partitions group exact peptides only;
  homologous peptides may span pool/dev/test.
- The 6 remaining flagged comments and all non-candidate rows (4,790)
  are excluded; their disposition is preserved in
  `data/processed/excluded_records.csv`.
