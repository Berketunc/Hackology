# chappy.md completion — 4 October 2026

- [x] Current README/handoff checkpoint committed before patching: `4f99482`.
- [x] Cached `esm2_mean_nn` and `esm2_joint_nn` added with the existing MLP procedure.
- [x] Original v2 design preserved; follow-up design separately locked before fitting.
- [x] All 13 mean and 13 joint full-budget jobs completed, in separate processes.
- [x] Training hashes, test rows, eligibility, null alpha, integer best epoch, and unchanged original jobs verified by `scripts/check_mlp_extension.py`.
- [x] Paired contrasts computed for all four regimes; strict uses a fixed-split paired-allele bootstrap, and two-fold locus intervals remain descriptive.
- [x] Headline/report rewritten around representation × fitting procedure; original P1–P4 and key audits retained.
- [x] Linear-head offset invariance explained with the qualification that independently fitted models need not have identical peptide coefficients.
- [x] Locus results demoted to an appendix.
- [x] Noise-ceiling explanation expanded and primary papers checked; source review records the figure-image retrieval limitation. No unverified numerical repeatability coefficient or dataset ceiling was asserted.
- [x] IEDB pilot archived, runbooks moved into notes, optional website relocated with documented launcher and verified routes.
- [x] Uncommitted main-benchmark arrays and cache/re-extraction requirements documented.
- [x] 17 Python tests pass. No new GPU extraction or optional model family was started.

Outcome: matching the MLP procedure improves ESM-2, but does not reverse its disadvantage to BLOSUM in either primary regime. The report does not claim a causal allocation to head capacity alone: preprocessing, optimization, feature dimension, and parameter count also matter.

The runbook's proposed exact identity across separate Ridge fits and its phrase “the only arm that goes positive under locus transfer” were not copied as established facts. Offset invariance holds within a fitted additive model; fitted coefficients can differ, and the original locus deltas include small positive values for other approaches as well.
