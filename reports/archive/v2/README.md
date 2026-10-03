# pHLASP — Peptide–HLA Stability Predictor

**Find your epitope.** A reproducible research benchmark and local web workspace for predicting peptide–HLA complex stability, scanning proteins for candidate 9-mers, and comparing sequence baselines with frozen protein language model representations.

**Current finding:** the sequence-aware BLOSUM MLP outperforms both tested ESM-2 representations on the main held-out-peptide and held-out-allele evaluations. Joint ESM-2 encoding shows an exploratory signal when an entire HLA locus is held out, but the evidence does not establish a reliable general advantage from pretraining.

This repository contains the prepared data, locked splits, predictions, statistical comparisons, compute measurements, figures, a working website with its opening animation, and an earlier active-learning study. The website uses real fitted models.

## Research question

> Under what conditions, if any, do general protein foundation model representations improve peptide–HLA stability prediction over sequence-aware supervised models trained on the same sequences, and what does that cost in compute?

A **peptide** is a short amino-acid sequence; an **HLA allele** identifies a variant of the protein that displays it. Here the prediction target is the measured dissociation **half-life in hours** of a peptide–HLA complex: how long it remains stable under the assay conditions. This is different from binding affinity or an immune response. The app helps explore candidate stability; a high prediction does not establish that a peptide will be an effective epitope.

**H1 — pretraining supplies an inductive bias.** Both approaches receive the same peptide and HLA sequence information. Once pretrained weights are fixed, a deterministic encoding cannot create additional label information beyond those inputs. Pretraining on large-scale protein sequence corpora may nevertheless help a predictor learn from limited labels. Greater benefit under label scarcity or greater sequence distance is a hypothesis, not a mathematical consequence of that observation.

| Prediction | What we tested | What we found |
|---|---|---|
| **P1:** little or no pLM advantage on data-rich held-out peptides | Five peptide-group folds at full label budget | Consistent with P1: both ESM-2 approaches lose to the sequence-aware MLP. |
| **P2:** an advantage on unseen alleles, increasing with distance from training alleles | Five allele-group folds, distance strata, and whole-locus holdouts | No advantage in the primary allele regime. The disadvantage narrows with distance; HLA-A holdout gives an exploratory joint-encoding signal. |
| **P3:** the pLM-minus-baseline gap decreases as the label budget grows | Nested 10%, 25%, 50%, and 100% training budgets | Fold-mean gaps decrease across the grid, but are already negative at 10%. No positive average benefit is demonstrated. |
| **P4:** extraction matters more than model choice | Independent mean pooling versus joint sequence encoding using the same ESM-2 checkpoint | Extraction affects ranking performance. The comparison with model choice remains untested because a second pLM was not run. |

See the [full findings and hypothesis assessments](reports/benchmark_findings.md).

## What is implemented

| Component | Current state | Evidence / entry point |
|---|---|---|
| Data preparation | Checksummed main dataset with zero labels retained, engineered constructs excluded, and per-allele audits | [Data manifest](data/manifests/rasmussen_manifest.json), [data loader](src/data.py) |
| Benchmark | Five approaches, four evaluation regimes, label-budget experiments; 215 model/fold/budget jobs | [Locked design](results/benchmark/design.json), [runner](src/run_benchmark.py) |
| Analysis | Macro and pooled metrics, paired confidence intervals, distance/support analysis, strict control, tier AUC, compute accounting | [Findings](reports/benchmark_findings.md), [result table](reports/benchmark_table.md) |
| Website | Protein scanning, single-pair prediction, batch CSV, model comparison, shortlists, saved runs, exports, and methods | [Frontend](web/), [implementation notes](web/README.md) |
| Animation | pHLASP helix/residue intro, moving 9-mer window, Skip/Escape, replay, and reduced-motion behavior | [Intro preview](reports/workspace_intro.png), [checks](results/benchmark/intro_checks.json) |
| Verification | Python tests, clean-clone baseline reproduction, live demo checks, desktop/mobile browser checks | [Validation section](#validation) |
| Earlier study | IEDB source review, predictor pilots, and random versus diversity-based acquisition | [Active-learning report](reports/al_comparison.md) |

## Quick start: run the website

Python **3.11** was used. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_benchmark --arms blosum_nn blosum_ridge onehot_ridge
python app.py
```

Open **http://127.0.0.1:7860**. The baseline command creates the fitted models needed for CPU-only scanning, batch scoring, and the three sequence-baseline comparisons. Saved fits resume. Model binaries are not committed, so this fitting step is needed on a fresh clone. Use comparisons with ESM-2 disabled until you have run the full benchmark setup below.

The frontend is plain HTML/CSS/JavaScript with FastAPI and vectorized Python inference. No Node installation or frontend build is required. The original Gradio interface remains at **http://127.0.0.1:7860/legacy**.

Interactive API documentation is at **http://127.0.0.1:7860/api/docs**. The local API exposes dataset metadata through `GET /api/bootstrap` and scoring through `POST /api/scan`, `/api/predict`, `/api/batch`, and `/api/compare`.

## Website features

The workspace implements the supplied website outline and subsequent animated layout, using the pHLASP branding, steel-blue design, compact panels, and responsive desktop/mobile layouts.

| Workspace page | What it does |
|---|---|
| **Protein scan** | Accepts one protein sequence or single FASTA record, 9–1,000 residues, and one to six of the 72 dataset alleles. Scores every overlapping 9-mer, preserves its 1-based position, and provides rankings, stability-tier filters, and a clickable window × allele heatmap. Identical pairs share inference work. |
| **Single prediction** | Compares a peptide–allele pair across the three sequence baselines and, optionally, both ESM-2 approaches. Shows dataset measurement counts, recorded zero fraction, a nearest better-measured allele, and audited SPEARMINT split membership. |
| **Batch prediction** | Accepts pasted or uploaded CSV with `peptide,allele` columns, up to 200 rows. Rejects invalid batches before scoring with row-specific feedback. |
| **Model comparison** | Compares up to 30 unique shortlisted peptide–allele pairs across the sequence baselines or all five models. |
| **Saved runs** | Saves scan/batch inputs, results, and shortlists in this browser's local storage, with restore and delete actions. Clearing browser data removes them. |
| **Methods** | Explains the models, target, displayed ranges, dataset coverage, and limitations. |

Rankings, batches, shortlists, and comparisons can be exported as CSV. Scans and batches use the fast sequence baselines. New ESM-2 inputs can require a public checkpoint download and several minutes of local inference on first use; cached dataset examples are faster. Measurement counts describe this dataset, not human population frequencies. The exposure audit distinguishes published training files from validation/test files; it does not establish absence from general protein-model pretraining.

The opening animation draws helices, docks 16 residues, moves a 9-mer window, and reveals the tagline before fading out after approximately **4.75 seconds**. Skip and Escape dismiss it immediately; Replay is available in the footer. Reduced-motion preferences bypass automatic playback, and explicit replay displays a static frame. Keyboard focus is restored on dismissal. Model loading runs independently. The animation is a schematic illustration, not a predicted molecular structure.

![pHLASP desktop workspace](reports/workspace_desktop.png)

More previews: [single prediction](reports/workspace_single.png), [mobile workspace](reports/workspace_mobile.png), [opening animation](reports/workspace_intro.png), [mobile animation](reports/workspace_intro_mobile.png).

### How to interpret an individual prediction

Models predict `log1p(half_life_hours)`; the display converts back to hours and clips negative predictions to zero. The stability tiers are **<2 h**, **2 to <6 h**, and **≥6 h**.

For any previously measured peptide, the new workspace uses its held-out-peptide-fold model, including when the requested allele pairing is new. For an entirely new peptide, it averages the five models' log-scale predictions before converting to hours.

Displayed ranges are **uncalibrated min–max ranges across five fitted models**. They describe sensitivity to training splits; they are not 95% confidence or prediction intervals. For a measured peptide, four of those five fits may have seen it. The previously added conformal calibration feature was reverted and is not part of the current app. Benchmark confidence intervals below describe performance differences across evaluation units, not certainty about a particular peptide's half-life.

## Dataset and target

The main benchmark uses the organizer-supplied **Rasmussen dataset**. Provenance, checksums, and the source-sheet URL are recorded in the [manifest](data/manifests/rasmussen_manifest.json).

| Stage | Peptide–HLA pairs | Allele labels | Recorded zeros |
|---|---:|---:|---:|
| Original organizer table | 28,166 | 75 | 5,679 |
| Main benchmark, after excluding engineered C67S constructs | 27,031 | 72 | 4,711 |

The exclusion removes 1,135 rows across three engineered allele labels. All main-set peptides have nine residues. HLA inputs use the **34-residue contact pseudosequence**; the available 182-residue groove sequence is an unrun alternative. The dataset contains HLA-A and HLA-B, with no HLA-C measurements.

The regression target is **`log1p(thalf_hours)`**. Zero measurements remain in the main analysis and per-allele zero fractions are reported. Zeros may represent left-censoring but are currently treated as exact labels. Input checks enforce sequence lengths, required values, and uniqueness of peptide–allele pairs.

The data loader uses the original organizer CSV when present, or the committed checksummed [processed source table](data/processed/rasmussen_all.csv) otherwise. The large raw IEDB export and external SPEARMINT CSVs are not needed for this benchmark or the saved demo exposure audit.

## Models and representations

A **model arm** simply means one approach being compared in the experiment.

| Approach / code name | Input representation | Predictor and role |
|---|---|---|
| **BLOSUM MLP** / `blosum_nn` | Positional BLOSUM62 encoding of the 9-mer and 34-residue HLA pseudosequence: 860 features | 256/64 ReLU neural network; the sequence-aware reference |
| **BLOSUM Ridge** / `blosum_ridge` | The same 860 sequence features | Common Ridge procedure; linear sequence control |
| **Allele-ID Ridge** / `onehot_ridge` | BLOSUM62 peptide features plus training-fitted allele one-hot features | Illustrative floor; an unseen allele has an all-zero allele-ID vector |
| **ESM-2 mean** / `esm2_mean` | Concatenated, separately mean-pooled peptide and HLA hidden states: 2,560 features | Common Ridge procedure on frozen protein representations |
| **ESM-2 joint** / `esm2_joint` | Concatenated hidden states at the nine peptide positions from `peptide + GGGG + pseudosequence`: 11,520 features | Common Ridge procedure on frozen joint sequence encoding |

Both ESM-2 approaches use **`facebook/esm2_t33_650M_UR50D`**, with 1,280 hidden dimensions. Joint encoding uses a synthetic sequence; it does not assert that attention models a physical peptide–HLA complex. No SPEARMINT/MINT stability-trained embeddings or NetMHCstabpan predictions are used as benchmark inputs or comparators.

The common Ridge procedure is **StandardScaler → randomized PCA (up to 256 components) → RidgeCV**. Alpha is chosen independently for each approach/fold/budget over 0.01–100,000 using training-only leave-one-out selection. The procedure is shared; the chosen alpha is not forced to be the same. Full-budget ESM-2 selections were 1,000 for mean pooling and 10,000 for joint encoding; sequence controls selected 0.01–100.

Outer test rows are excluded from all preprocessing and fitting. Scaling and PCA are not refitted inside each internal leave-one-out alpha-selection step, which is a limitation of the selection procedure. The MLP uses AdamW and a grouped validation subset of outer training data for early stopping: peptide groups for peptide holdouts, allele groups otherwise. Split seed is 0; model, PCA, inner-validation, and nested-subset seeds equal the outer fold number. Embedding caches validate checkpoint, dimensions, row identity, dataset checksum, and array availability.

## Experimental design

| Regime | Question | Split |
|---|---|---|
| **Peptide** | Can the predictor rank peptides it has not seen? | Five peptide-group folds; all rows for a peptide remain together |
| **Allele** | Can it transfer to an unseen HLA sequence? | Five allele-group folds; cross-allele peptide sharing is allowed |
| **Locus** | Can it transfer across a larger HLA sequence shift? | Train B/test A, and train A/test B |
| **Strict allele** | What happens when shared peptides are also removed? | Allele fold 0, excluding every training peptide present in its test set |

The strict control reduces training from **22,879 to 7,794 rows**, removing 15,085 shared-peptide rows while preserving **4,152 test rows** and 12 macro-eligible alleles. It combines removal of peptide sharing with a large reduction in label budget, so its performance change does not isolate a causal effect of sharing.

Peptide and allele regimes use nested **10%, 25%, 50%, and 100%** training budgets. Locus and strict controls use their full retained training sets. Every held-out allele has **zero same-allele training measurements**; support from a nearest training allele is a separate quantity.

The current v2 analysis contains **215 model/fold/budget jobs**: 200 original fits reused and re-scored, plus 15 new full-budget locus/strict fits. The v2 split and benchmark designs were locked before those new fits. Original peptide/allele splits and 270,310 primary-regime full-budget predictions remain unchanged. Re-scoring earlier results under the revised primary metric is a **retrospective reanalysis**, not independent confirmation. Designs and outputs are retained in [split records](results/splits/) and [benchmark records](results/benchmark/); v1 is preserved under [results/archive](results/archive/) and [reports/archive/v1](reports/archive/v1/).

### Metrics and confidence intervals

The primary endpoint is **macro-averaged within-allele Spearman correlation**: rank peptides within each eligible allele, then weight eligible alleles equally. Eligibility is evaluated separately within each test fold and requires **at least 20 test rows and 10 distinct positive recorded half-lives**. Zero labels remain in the correlation. Constant predictions for an eligible allele make the macro undefined instead of silently dropping that allele. [Eligibility audits](results/benchmark/allele_metric_audit.csv) and [exclusions](results/benchmark/excluded_alleles.csv) include absent alleles with zero test rows.

Pooled Spearman, pooled-minus-macro, Pearson, log-scale RMSE, and tier AUC at 2/6 hours are secondary diagnostics. Pooled-minus-macro is a descriptive aggregation gap, not a causal decomposition of allele offsets. For alleles with at least 50% zeros—HLA-B*41:01, B*45:01, B*51:01, and B*55:01—lead with [tier AUC](results/benchmark/high_zero_summary.csv). Thresholds are compared on the log scale to preserve tier boundaries.

- **Peptide/allele comparisons:** paired 95% t intervals over five fold-level differences. The “±” beside a score is fold standard deviation, not its confidence interval.
- **Locus summary:** a two-fold t interval, which is highly unstable.
- **Per-locus, strict, distance, and support comparisons:** paired-allele bootstrap intervals conditional on the fixed split, using 10,000 resamples and seed 0.

These intervals are nominal and exploratory. Shared training data, related alleles, multiple comparisons, and method selection are not fully captured; bootstrap intervals do not represent repeated independent experiments.

## Results

All tables below use the **full training budget**. Positive Δ favors the named approach over the MLP. All five models are included, including controls that perform poorly.

### Held-out peptides — five folds

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.612 ± 0.013 | reference | 0.753 | +0.141 |
| BLOSUM Ridge | 0.258 ± 0.022 | -0.354 [-0.385, -0.323] | 0.556 | +0.299 |
| Allele-ID Ridge (floor) | 0.258 ± 0.022 | -0.354 [-0.384, -0.323] | 0.559 | +0.301 |
| ESM-2 mean | 0.168 ± 0.021 | -0.443 [-0.471, -0.416] | 0.523 | +0.354 |
| ESM-2 joint | 0.215 ± 0.029 | -0.396 [-0.424, -0.369] | 0.406 | +0.190 |

### Held-out alleles — five folds

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.392 ± 0.126 | reference | 0.473 | +0.080 |
| BLOSUM Ridge | 0.228 ± 0.073 | -0.164 [-0.240, -0.089] | 0.246 | +0.018 |
| Allele-ID Ridge (floor) | 0.228 ± 0.073 | -0.164 [-0.240, -0.089] | 0.210 | -0.018 |
| ESM-2 mean | 0.160 ± 0.067 | -0.233 [-0.322, -0.143] | 0.292 | +0.132 |
| ESM-2 joint | 0.190 ± 0.081 | -0.202 [-0.274, -0.131] | 0.268 | +0.078 |

### Entire-locus transfer — two folds

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.062 ± 0.012 | reference | 0.126 | +0.063 |
| BLOSUM Ridge | 0.071 ± 0.039 | +0.009 [-0.443, +0.461] | 0.101 | +0.030 |
| Allele-ID Ridge (floor) | 0.071 ± 0.038 | +0.009 [-0.438, +0.456] | 0.088 | +0.017 |
| ESM-2 mean | 0.071 ± 0.003 | +0.008 [-0.122, +0.139] | 0.142 | +0.072 |
| ESM-2 joint | 0.112 ± 0.019 | +0.050 [-0.226, +0.325] | 0.013 | -0.099 |

### Strict allele holdout — fold 0 only

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.147 | reference | 0.215 | +0.069 |
| BLOSUM Ridge | 0.080 | -0.066 [-0.191, +0.066] | 0.124 | +0.044 |
| Allele-ID Ridge (floor) | 0.083 | -0.064 [-0.188, +0.068] | 0.081 | -0.002 |
| ESM-2 mean | -0.024 | -0.170 [-0.273, -0.061] | -0.010 | +0.014 |
| ESM-2 joint | 0.009 | -0.138 [-0.256, -0.005] | 0.056 | +0.048 |

Peptide/allele: paired t intervals over five folds. Locus summary: two-fold t interval, highly unstable; see per-locus paired-allele intervals. Strict: paired-allele bootstrap conditional on fold 0. Intervals do not account for shared training data or method selection.

### Interpretation

The MLP leads in both primary evaluation regimes. Joint encoding improves over independent means by **+0.047 [95% CI +0.027, +0.067]** on peptide holdout and **+0.031 [−0.002, +0.063]** on allele holdout; the second interval includes zero.

When HLA-A is entirely held out, joint ESM-2 scores **0.125** versus the MLP's **0.054**, a paired-allele difference of **+0.071 [+0.004, +0.133]**. For HLA-B holdout the difference is **+0.028 [−0.029, +0.086]**. These are exploratory, split-conditional results on 34 eligible alleles per locus, with low absolute correlations. The overall two-locus interval does not establish a general advantage.

At 10% of the allele-training budget, mean-minus-MLP is **−0.057 [−0.121, +0.007]**, and joint-minus-MLP is **−0.028 [−0.087, +0.030]**. Their intervals include zero; the label-budget trend does not demonstrate a positive benefit at small budgets.

**H1 remains plausible, but a reliable general benefit is not established.** The main finding is specific to these representations, predictors, labels, and evaluation regimes. Full interpretation, eligibility exclusions, and strict-versus-permissive comparisons are in the [findings](reports/benchmark_findings.md).

![Macro ranking, paired differences, and locus stress tests](reports/allele_distance.png)

![Budget differences with paired confidence intervals](reports/learning_gap.png)

The [absolute learning curves](reports/learning_curve.png) complement the difference plot.

## Compute cost

Recorded costs from [compute.csv](results/benchmark/compute.csv):

| Approach | One-time extraction GPU-function seconds | Peak allocated GPU memory, GB | CPU fit/predict seconds, all full-budget regimes |
|---|---:|---:|---:|
| BLOSUM MLP | 0 | — | 31.70 |
| BLOSUM Ridge | 0 | — | 9.68 |
| Allele-ID Ridge | 0 | — | 47.06 |
| ESM-2 mean | 23.40 | 1.43 | 28.84 |
| ESM-2 joint | 83.88 | 1.48 | 223.16 |

Mean extraction processes 5,705 deduplicated individual sequences, at approximately 623 sequences/s during inference. Joint extraction processes 27,031 pair sequences, at approximately 338 sequences/s. These workloads and counting units differ. GPU-function timing includes worker model loading but excludes container startup, transfer, and a full billing total. CPU timings cover the full-budget benchmark fits, not website request latency or all label-budget fits. They are measured run timings, not hardware-independent promises.

The v2 analysis reused existing embeddings and required **no additional GPU extraction**. GPU seconds are not total financial cost.

![Macro ranking versus extraction GPU time](reports/compute_performance.png)

## Reproduce the full benchmark

After installing the requirements as in Quick start:

```sh
# Authenticate Modal if embedding extraction is needed.
modal setup

# All five models, four regimes, full training budget.
python -m src.run_benchmark

# Include the 10/25/50/100% label-budget experiments.
python -m src.run_benchmark --fractions .1 .25 .5 1

# Regenerate tables and figures from saved results.
python -m src.report
```

Missing embedding binaries trigger Modal GPU extraction, requiring network access, an authenticated Modal account, and GPU billing. The public ESM-2 checkpoint does not require a Hugging Face secret. Saved fits resume, while missing model artifacts on a fresh clone are rebuilt. Large feature/model binaries are ignored; processed data, result CSVs, and provenance are committed.

To force refitting, move saved jobs/models aside in a disposable checkout. Do not mix outputs from different experimental designs; `design.json` guards against incompatible configurations. The CPU-only command in Quick start avoids Modal and reconstructs all 39 full-budget sequence-baseline fits across the four regimes.

### Result files

| File / directory | Contents |
|---|---|
| [per_row.csv](results/benchmark/per_row.csv) | 426,225 full-budget prediction rows across models and regimes |
| [per_fold.csv](results/benchmark/per_fold.csv) | Metrics for **all budgets**; filter `fraction == 1` for full-budget results |
| [per_allele.csv](results/benchmark/per_allele.csv) | Per-allele evaluation metrics |
| [paired_comparisons.csv](results/benchmark/paired_comparisons.csv) | Model-minus-reference differences and paired intervals |
| [distance_stratified.csv](results/benchmark/distance_stratified.csv) | Distance and support stratification |
| [locus_paired_ci.csv](results/benchmark/locus_paired_ci.csv) | Per-locus paired-allele comparisons |
| [strict_vs_permissive.csv](results/benchmark/strict_vs_permissive.csv) | Strict peptide-sharing control |
| [fit_compute.csv](results/benchmark/fit_compute.csv) | Fit metadata, selected hyperparameters, timing, and reuse provenance |
| [jobs/](results/benchmark/jobs/) | Per-fit raw predictions and metadata, including smaller budgets |

## Validation

The current Python suite has **17 passing tests** covering split boundaries, zero handling, macro eligibility and weighting, paired-CI alignment, training-only preprocessing, cache checks, the corrected IEDB audit, input validation, window positions, tier boundaries, agreement with saved predictions, and held-out routing of novel allele pairings.

```sh
python -m pytest -q
```

Create the baseline model artifacts before running workspace prediction tests. For full live checks, prepare all five models and keep `python app.py` running in a separate terminal:

```sh
python -m scripts.check_demo
pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/check_workspace.py
python scripts/check_intro.py
```

Recorded verification:

- [Benchmark validation](results/benchmark/validation.json): 215 jobs and 426,225 finite full-budget predictions, with original primary predictions unchanged.
- [Clean-clone reproduction](results/benchmark/reproducibility.json): at benchmark commit `805fad9`, a fresh Python 3.11 environment without the original organizer file reproduced all **39 baseline fits / 255,735 predictions with maximum absolute difference 0.0**. Regenerated comparison tables matched exactly; the then-current 12 tests passed. Full ESM-2 extraction was not repeated in that clone. This is a benchmark reproduction check, not a clean-clone verification of subsequent website changes.
- [Live legacy demo](results/benchmark/demo_checks.json): both example inputs return all five approaches.
- [Workspace browser checks](results/benchmark/workspace_checks.json): real 62-window × six-allele scan, filtering, heatmap, pair inspection, shortlists, CSV upload/export, saved-run restoration, five-model predictions, invalid-input feedback, and mobile layout. Passed in isolated headless Chromium with no browser errors.
- [Animation checks](results/benchmark/intro_checks.json): residue/window alignment, tagline timing, automatic exit, replay, Skip/Escape, focus restoration, portrait/landscape layouts, and reduced motion. Passed with no browser errors. Desktop and mobile screenshots are available above.

## Earlier IEDB active-learning study

The repository also retains the initial research question: **can representation-based acquisition choose stability measurements that improve prediction faster than random selection?** That study uses a distinct source-reviewed set of **5,815 positive peptide–HLA pairs across 10 alleles** and its own fixed pool/development/final-test partitions. Its final test remains untouched.

The source review, overlap audit, predictor pilots, and three-policy comparison are complete. Random acquisition beat both sequence-diversity and embedding-diversity acquisition at every tested budget in all 10 seeds. This is a negative result for those pure-diversity policies with that predictor. Feature geometry is a possible explanation, not an established cause; uncertainty-based acquisition has not been tested, so the result does not show that active learning generally fails.

The corrected IEDB comment audit distinguishes the phrase “at least two independent experiments” from bounded outcome measurements. It leaves 6,101 rows without automatic flags, including the 5,815 positive candidates, and six flagged comments. Passing the screen alone does not establish source verification. The subsequent review and its assumptions are documented separately.

Read the [source review](reports/source_review.md), [overlap audit](reports/overlap_report.json), [pilot findings](reports/pilot_findings.md), and [active-learning comparison](reports/al_comparison.md). Original scripts and result directories remain available; these results should not be mixed with the main 27,031-pair benchmark.

## Limitations and work not yet done

- **Data scope:** nine-residue peptides, class I HLA-A/B, one dataset, and a binder-enriched measurement setting. Counts measure dataset coverage, not population representation. Whole-protein scanning has not been independently validated on a new protein distribution.
- **Measurement uncertainty:** zeros may be censored. Source descriptions report averages of at least two experiments, but per-replicate variance is unavailable, so the assay-noise ceiling is unestimated.
- **Generalization:** ordinary allele splits permit peptide sharing; near-sequence similarity is not clustered. The strict control uses one fold and also reduces training size. Only two locus holdouts are available.
- **Model scope:** one frozen pLM, two extraction strategies, one common Ridge/PCA procedure, and a separate MLP reference. The pseudosequence and synthetic linker may be out of distribution for the pretrained model.
- **Uncertainty:** benchmark intervals have the dependence and selection limitations stated above. The website has no calibrated per-prediction interval or probability of correctness.
- **Unrun extensions:** zero-shot masked likelihood, a second pLM, the 182-residue groove representation, structural analysis, and broader pretraining-contamination analysis. Uncertainty-based active learning also remains unrun.
- **Claims:** no demonstrated laboratory savings, therapeutic efficacy, immunogenicity, clinical benefit, or numerical noise ceiling is claimed.

The [benchmark implementation record](reports/implementation_status.md) documents the fix2 scope and its historical validation; [website notes](web/README.md) cover the later interface and animation.

## Repository map

| Path | Purpose |
|---|---|
| [app.py](app.py) | Local website entry point, with legacy Gradio mounted alongside it |
| [src/](src/) | Main data, representations, models, splits, evaluation, reporting, and website inference/API |
| [web/](web/) | Website HTML, styles, JavaScript, animation, and supplied design-system documentation |
| [scripts/](scripts/) | Embedding extraction, earlier IEDB work, and live/browser verification |
| [tests/](tests/) | Benchmark and workspace regression checks |
| [data/processed/](data/processed/) | Prepared tables and locally generated feature caches |
| [data/manifests/](data/manifests/) | Input provenance and checksums |
| [results/benchmark/](results/benchmark/) | Current v2 predictions, metrics, fit records, and verification evidence |
| [results/splits/](results/splits/) | Locked evaluation assignments and split design |
| [reports/](reports/) | Scientific findings, plots, audits, website screenshots, and implementation notes |
| [requirements.txt](requirements.txt) / [requirements-dev.txt](requirements-dev.txt) | Pinned runtime dependencies and browser-check dependencies |

Method references: [RidgeCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html), [paired t-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).
