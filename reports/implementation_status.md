# Current implementation status

The supplied-data benchmark now has seven approaches and 241 model/fold/budget jobs. The original v2 benchmark (five approaches, 215 jobs) is unchanged at the per-job level. The matched-head extension adds 26 full-budget ESM-2 + MLP fits, with no new GPU extraction.

- The original `design.json` and all split files remain byte-identical. `mlp_extension_design.json` records the 26-fit follow-up separately, before fitting.
- Both new approaches use the same `fit_baseline_nn` implementation and grouped inner validation as BLOSUM. Mean and joint inputs reuse their validated feature caches; no PCA was added to the MLP. Input dimensions differ, so parameter counts are not matched.
- All 26 JSON records have null alpha, an integer best epoch, and training-row hashes matching the reference. Test-row identities and eligible-allele counts match, too. There are 596,715 finite full-budget predictions.
- Reports include seven-approach tables, matched-head contrasts with paired intervals, updated figures, and shared-extraction compute accounting. The headline concerns representation and fitting procedure; locus transfer is in an exploratory appendix.
- The noise-ceiling explanation identifies missing replicate values and unique allele–peptide rows. A separate review cites primary assay evidence without inventing a dataset-specific ceiling.
- Historical IEDB scripts, prepared data, results, reports, and the existing tracked embedding binary are under `archive/iedb_pilot/`. No archived result was recomputed; its final test remains untouched. Historical paths in those reports/manifests are preserved as provenance.
- Runbooks and handoff notes are under `notes/`. The optional website launcher and FastAPI wrapper are under `notes/website/`; launch with `python -m notes.website.app`. The website retains its original five predictors and animation.

Matched-head validation: 17 Python tests passed after relocation. The workspace, metadata API, API documentation, and mounted Gradio routes return HTTP 200 using the relocated application. `python -m scripts.check_mlp_extension` passes all acceptance checks and verifies all 215 original jobs, splits, and the original design against checkpoint `4f99482`. Active documentation links and table structure were checked; updated scientific figures were inspected.

Historical clean-clone validation at `805fad9` remains in `results/benchmark/reproducibility.json`; it covered the original CPU baselines, not this extension. Historical browser evidence remains in `workspace_checks.json` and `intro_checks.json`; the frontend assets were unchanged in this follow-up. See [v2 history](archive/v2/implementation_status.md), [current findings](benchmark_findings.md), and [acceptance evidence](../results/benchmark/mlp_extension_checks.json).

## Post-hoc patience sensitivity (checkv1)

Default patience remains 15. A separate 18-fit study uses patience 30 for all three MLP approaches on the allele and strict regimes; it does not overwrite any main benchmark job, summary, design, split, or headline table. 5/18 fits still select epoch ≤2 and no further tuning is performed. The full-budget epoch appendix and paired comparisons are in [patience_sensitivity.md](patience_sensitivity.md).

Current validation: **18 Python tests pass**, including refitting the original BLOSUM allele-fold-0 model with the unchanged default. The sensitivity verifies all protected benchmark hashes, training-row hashes, test rows, and macro eligibility. Seven-approach compute and matched-head figures regenerate in an isolated directory, with exact matched-head table agreement. Input truncation and structure-model scope are now explicit, and the user-supplied source-sheet verification is retained as an attributed receipt.
