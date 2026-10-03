# Implementation against fix.md

- Phase 0: portable configuration, explicit dependency versions, visible results/processed CSVs, removed duplicate extractor and tracked junk. Historical script changes are limited to path portability, cache validation, and the documented vectorizer leakage correction. Existing results are preserved, not silently regenerated.
- Phases 1–2: validated organiser dataset, retained zeros, excluded engineered alleles, checksummed derivatives, locked peptide/allele folds, allele-distance table, BLOSUM MLP and matched Ridge baselines, ranking and tier metrics.
- Phase 3 core: Arms A and B, identical Ridge pipeline, cached general ESM-2 features with checkpoint/dimension/row identity checks, Modal timing and peak memory. Optional C/D/E are not included in this minimum submission.
- Phase 4: five-fold scores, per-allele/identity/support stratification, 10/25/50/100% learning curves, compute and findings report.
- Phase 5: local Gradio demo, model comparison, measured-count context, better-measured neighbour, overlap flags, two examples. Known pairs show predictions from the fold that excluded their peptide. The five-fit range is explicitly not a calibrated interval; other fits can include the pair.
- Phase 6.3: corrected audit merged after the live demo worked; regression verifies 6,101 unflagged / 5,815 candidates / 6 comments. No structure prediction or supervised-checkpoint contamination experiment.
- Phase 7: README, committed tables and derivatives, clean-copy checks. Large feature caches and fitted models are regenerated, not committed.

Interpretation corrections: held-out alleles have zero same-allele training measurements, so stratification uses nearest-training-allele support as a separately named variable. Sequence identity is not an independent measure of generalization difficulty. Synthetic concatenation is not a structural model. P4's between-model comparison cannot be tested without optional Arm D. The old active-learning loss is established; feature geometry as its cause remains a hypothesis.
