# Implementation against fix2.md

- Phase 0 was already complete: tracked processed/results CSVs, portable paths, pinned dependencies, removed tracked junk and conflicting extractor.
- Phase 1 adds locus, per-allele zeros, locked A/B locus holdouts, distance/zero/count tables, and strict allele fold 0. No HLA-C rows exist. Existing peptide and allele assignments are byte-identical to v1.
- Phase 2 changes the primary endpoint to macro within-allele Spearman with ≥20 test rows and ≥10 distinct positive labels. Counts and reasons are exported, including absent test alleles. Pooled correlation and pooled-minus-macro are diagnostics. Fold-paired t CIs and paired-allele bootstrap CIs are implemented. Ridge was already tuned per arm; its selection records are now explicit in the report.
- Phase 3 retains the two core frozen ESM-2 arms and renames Arm B to joint sequence encoding throughout active code, results and UI. No inference that attention represents a physical complex. Cached arrays were renamed and validated, not re-extracted. Timing, peak memory and both inference/function throughput are recorded.
- Phase 4 refreshes the main tables, primary figure with locus anchors, per-distance/support paired CIs, label-budget gaps with CIs, macro/compute figure, strict control and findings. High-zero alleles are named with tier AUCs. Unknown assay noise ceiling is stated.
- Phase 5 retains the local demo, both examples, actual training-fold row counts, nearest better-measured allele, overlap audit and explicit uncertainty limitations. Counts are labelled as belonging to this dataset.
- Phase 6.3 was already complete and is regression-tested. Other stretch tasks and optional Arms C/D/E remain deferred under the minimum-viable scope.
- Phase 7 updates README, versions outputs and verifies tests, data fallback and live examples. v1 results/splits/reports are preserved under `results/archive/` and `reports/archive/v1/`.

There are 215 fitted-model/budget jobs: 200 reused original fits and 15 new full-budget fits (five arms × two locus holdouts plus one strict fold). New fits began only after v2 split and benchmark designs were written. Re-scoring original predictions under the user's revised metric is explicitly retrospective. The old IEDB scripts and results were not changed for v2.

Verification: all 12 tests pass in a fresh Python environment and a clean clone of `805fad9`. All 39 full-budget sequence-baseline fits reproduce exactly across 255,735 predictions using the committed data fallback. All 477 result CSVs and seven processed CSVs are tracked. Both live demo examples return all five arms; no browser visual check was available.
