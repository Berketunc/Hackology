# checkv1 completion — 4 October 2026

- [x] Clean-state checkpoint before changes: `2df2f32`.
- [x] `patience=15` parameter added, original default unchanged.
- [x] Separate design locked before fits; all original benchmark artifacts hashed and protected.
- [x] All 18 patience-30 fits completed for BLOSUM, mean ESM-2 and joint ESM-2 (five allele folds and one strict fold each).
- [x] Training hashes, test rows, eligible-allele counts and finite predictions verified; 5 early selected checkpoints remain and are explicitly reported.
- [x] Paired comparisons against BLOSUM at both patience values and within-approach sensitivity intervals written; main tables unchanged.
- [x] Every original full-budget MLP best epoch exposed in a per-fold appendix, alongside sensitivity epochs where available.
- [x] Actual 34-residue inputs distinguished from the supplied 182-residue α1/α2 fragment; missing native-chain/complex context disclosed.
- [x] Structure omission explained using resource scope and target mismatch, citing official Boltz documentation. No unmeasured GPU-day estimate asserted.
- [x] User-provided direct-sheet verification retained as an attributed receipt, not represented as a newly repeated live check.
- [x] Root cleanup and uncommitted-cache disclosure retained. Seven-model figures regenerate in isolation; matched-head table is identical.
- [x] 18 tests pass, including default-fit preservation. No GPU extraction, optional model, alternative validation split, patience 60, or matched-MLP budget curve was started.

Interpretation corrections: a best epoch of 1 does not mean an untrained model, persistent early selection at patience 30 does not identify a root cause, and an allele-only rerun cannot establish peptide-regime insensitivity. All three qualifications are explicit in the report.
