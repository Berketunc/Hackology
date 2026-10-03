# When does protein pretraining help peptide–HLA stability prediction?

The research question is whether general protein-model representations improve stability prediction over a **sequence-aware supervised model**, under what distribution shift or label scarcity, and at what compute cost. The headline endpoint is **macro-averaged within-allele Spearman**, not pooled correlation. The BLOSUM-pseudosequence neural network is the reference; allele-ID Ridge is an illustrative floor.

H1: both models receive the same peptide and HLA sequence information. With pretrained weights fixed, a deterministic encoding cannot create label information beyond its inputs; pretraining on large-scale protein sequence corpora supplies an inductive bias that may improve finite-sample learning. Greater benefit under distance or scarcity is a hypothesis, not a theorem implied by that observation.

- P1: little or no pLM advantage on data-rich held-out peptides.
- P2: an advantage over the BLOSUM-pseudosequence reference on held-out alleles, growing with distance to training alleles.
- P3: the advantage shrinks as the training-label budget grows.
- P4: extraction matters more than model choice. The extraction contrast is tested; a second pLM remains optional and unrun.

## Results

The MLP leads on ordinary peptide and allele holdouts. Joint ESM-2 encoding improves on independent mean pooling under the primary within-allele metric. Locus transfer is difficult for every arm; joint encoding has an exploratory advantage on HLA-A holdout, while the overall two-locus interval is too wide for a general claim. See [findings](reports/benchmark_findings.md) for paired CIs, eligibility exclusions, tier AUC, noise limitations and interpretation.

### peptide

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.612 ± 0.013 | reference | 0.753 | +0.141 |
| BLOSUM Ridge | 0.258 ± 0.022 | -0.354 [-0.385, -0.323] | 0.556 | +0.299 |
| Allele-ID Ridge (floor) | 0.258 ± 0.022 | -0.354 [-0.384, -0.323] | 0.559 | +0.301 |
| ESM-2 mean | 0.168 ± 0.021 | -0.443 [-0.471, -0.416] | 0.523 | +0.354 |
| ESM-2 joint | 0.215 ± 0.029 | -0.396 [-0.424, -0.369] | 0.406 | +0.190 |

### allele

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.392 ± 0.126 | reference | 0.473 | +0.080 |
| BLOSUM Ridge | 0.228 ± 0.073 | -0.164 [-0.240, -0.089] | 0.246 | +0.018 |
| Allele-ID Ridge (floor) | 0.228 ± 0.073 | -0.164 [-0.240, -0.089] | 0.210 | -0.018 |
| ESM-2 mean | 0.160 ± 0.067 | -0.233 [-0.322, -0.143] | 0.292 | +0.132 |
| ESM-2 joint | 0.190 ± 0.081 | -0.202 [-0.274, -0.131] | 0.268 | +0.078 |

### locus

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.062 ± 0.012 | reference | 0.126 | +0.063 |
| BLOSUM Ridge | 0.071 ± 0.039 | +0.009 [-0.443, +0.461] | 0.101 | +0.030 |
| Allele-ID Ridge (floor) | 0.071 ± 0.038 | +0.009 [-0.438, +0.456] | 0.088 | +0.017 |
| ESM-2 mean | 0.071 ± 0.003 | +0.008 [-0.122, +0.139] | 0.142 | +0.072 |
| ESM-2 joint | 0.112 ± 0.019 | +0.050 [-0.226, +0.325] | 0.013 | -0.099 |

### allele_strict

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.147 | reference | 0.215 | +0.069 |
| BLOSUM Ridge | 0.080 | -0.066 [-0.191, +0.066] | 0.124 | +0.044 |
| Allele-ID Ridge (floor) | 0.083 | -0.064 [-0.188, +0.068] | 0.081 | -0.002 |
| ESM-2 mean | -0.024 | -0.170 [-0.273, -0.061] | -0.010 | +0.014 |
| ESM-2 joint | 0.009 | -0.138 [-0.256, -0.005] | 0.056 | +0.048 |

Peptide/allele: paired t intervals over five folds. Locus summary: two-fold t interval, highly unstable; see per-locus paired-allele intervals. Strict: paired-allele bootstrap conditional on fold 0. Intervals do not account for shared training data or method selection.

The primary figure remains the distance-stratified result, including the locus stress tests, regardless of whether an overall difference excludes zero:

![Macro ranking, paired differences and locus stress tests](reports/allele_distance.png)
![Budget differences with paired confidence intervals](reports/learning_gap.png)
![Macro ranking versus extraction GPU time](reports/compute_performance.png)

## Reproduce

### Website workspace

Run `python app.py` and open **http://127.0.0.1:7860**. The workspace implements the supplied “Website outline Protein binding research tool” design: protein scanning, single-pair prediction, batch CSV upload, model comparison, saved runs and methods. The original Gradio interface remains at **http://127.0.0.1:7860/legacy**.

- Scan one protein or FASTA record (9–1,000 residues) against up to six of the 72 dataset alleles. Every overlapping 9-mer keeps its original 1-based position; identical pairs share inference work. Rankings, tier filters and the clickable window × allele map use the selected sequence baseline.
- Inspect selected pairs or single inputs across the three sequence baselines and, optionally, both ESM-2 arms. New ESM-2 inputs can require a public checkpoint download and several minutes of local inference on first use. Protein scans and batch ranking stay on the fast sequence baselines.
- Paste or upload up to 200 `peptide,allele` CSV rows. Invalid batches are rejected before inference, with row-specific feedback. Shortlist up to 30 unique peptide–allele pairs and compare their models side by side.
- Export rankings, batches, shortlists and comparisons to CSV. Saved scan/batch runs retain their inputs, results and shortlist in this browser's local storage; clearing browser data removes them. No synthetic example runs or fabricated scores are used.

All displayed ranges remain **uncalibrated five-fit min–max ranges**. The reverted conformal interval feature is not included. The new workspace routes any previously measured peptide to its held-out model, even for a novel allele pairing; entirely new peptides average the five log-scale predictions. Whole-protein scanning is exploratory and does not establish accuracy on a new protein distribution.

The frontend is plain HTML/CSS/JavaScript in `web/`, with a FastAPI backend in `src/webapp.py` and vectorized inference in `src/workspace.py`. No frontend build or Node dependency is needed. [Desktop screenshot](reports/workspace_desktop.png), [single-pair screenshot](reports/workspace_single.png), [mobile screenshot](reports/workspace_mobile.png).

Optional browser verification (while the app is running):

```sh
pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/check_workspace.py
```

This uses an isolated browser context and writes its checks to `results/benchmark/workspace_checks.json`. The Python suite also tests FASTA/CSV validation, window-position preservation, tier boundaries, real benchmark prediction agreement, and held-out routing of novel allele pairings.

### Benchmark reproduction

Python 3.11 was used. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_benchmark
python -m src.report
python app.py
```

The default benchmark evaluates all five core arms across peptide, allele, locus and strict allele-fold-0 regimes at full budget. Saved fits resume. On a fresh clone, missing fitted models are rebuilt. Missing embedding binaries trigger Modal extraction; this requires network access, an authenticated Modal account (`modal setup`) and GPU billing. No Hugging Face secret is required for this public checkpoint. Large binary features/models are not committed; processed tables, result CSVs and provenance are.

Without Modal, run the sequence baselines only:

```sh
python -m src.run_benchmark --arms blosum_nn blosum_ridge onehot_ridge
```

Full label-budget curves and checks:

```sh
python -m src.run_benchmark --fractions .1 .25 .5 1
python -m src.report
python -m pytest -q
python -m scripts.check_demo  # while app.py serves localhost:7860
```

Only peptide/allele regimes use the four budget fractions; the locus and strict checks use their full retained training sets. To force refits, move saved jobs/models aside in a disposable checkout. Do not mix results from different designs. `design.json` files reject incompatible changes.

The data layer reads the original organiser file when present, or the committed checksummed `data/processed/rasmussen_all.csv` otherwise. The huge IEDB raw export and external SPEARMINT CSVs are not needed to run the new benchmark or the saved demo exposure audit. Archived v1 results are in `results/archive/benchmark_v1/`; active results are v2. The original 200 model/fold/budget jobs were re-scored without changing their fits; 15 locus/strict jobs were newly fit after locking the v2 design. This is partly a retrospective metric reanalysis, not a new independent confirmation.

## Design and metric rules

The organiser data contain 28,166 pairs, 75 allele labels and 5,679 zeros. Excluding 1,135 engineered C67S rows leaves 27,031 pairs, 72 alleles and 4,711 zeros. All peptides have nine residues; primary HLA inputs are 34 contact residues. Per-allele zero counts/fractions are recorded. Only HLA-A and HLA-B occur: the locus stress tests train B/test A and train A/test B.

Five peptide-group folds provide the control. Five allele-group folds are the main test; peptide sharing across alleles is permitted. Strict allele fold 0 removes all training peptides that occur in its test set: 22,879 → 7,794 training rows, unchanged 4,152 test rows. That change combines removal of shared peptides with a lower label budget. Every held-out allele has zero same-allele training measurements; nearest-training-allele support is a separately labelled quantity. Counts describe this dataset, not human population representation.

Within each evaluation fold, an allele enters macro Spearman only with **≥20 test rows and ≥10 distinct positive half-lives**. Eligible alleles receive equal weight; zero labels remain in their correlations. Counts and exclusions appear in `allele_metric_audit.csv` and `excluded_alleles.csv`. A constant prediction for an eligible allele makes the macro undefined instead of silently dropping that allele. The main table averages per-fold macros; stratified analyses average eligible allele scores within the named stratum.

Pooled Spearman, pooled-minus-macro, log1p RMSE, Pearson and tier AUC at 2/6 h are diagnostics. The pooled-minus-macro gap is not a causal decomposition of between-allele offsets. For alleles with ≥50% recorded zeros, lead with tier AUC; the findings name them. Boundary thresholds are compared on the log scale to avoid floating-point round-trip misclassification.

Headline model differences use paired 95% t intervals across five folds. Per-locus, strict-fold and distance/support intervals bootstrap paired allele scores within the fixed split (seed 0, 10,000 resamples). These intervals are nominal and exploratory: shared training rows, related alleles, multiple comparisons and method selection are not fully represented. The two-locus t interval is especially imprecise.

## Models and representations

The sequence-aware reference is a 256/64 ReLU MLP using BLOSUM62 peptide/pseudosequence inputs, with grouped validation carved from outer training data for early stopping. Peptide groups are used for peptide-split validation; allele groups otherwise.

The common Ridge procedure is StandardScaler → randomized PCA (up to 256) → RidgeCV. **Alpha is selected separately for each arm/fold/budget** over 0.01–100,000 using training-only LOO. Architecture and selection procedure are shared, not the selected alpha. All preprocessing excludes outer test rows; transforms are not re-fit inside each internal LOO step. The MLP is a separate nonlinear reference. Split seed is 0; model, PCA, inner-validation and nested-subset seeds equal the outer fold number.

- ESM-2 independent mean: concatenate separately mean-pooled peptide and pseudosequence hidden states.
- ESM-2 joint sequence encoding: concatenate nine peptide-position hidden states from `peptide + GGGG + pseudosequence`.

Both use frozen `facebook/esm2_t33_650M_UR50D` (1,280 hidden dimensions). Joint encoding is a synthetic sequence input, not an assertion of physical interactions. Cache checkpoint, dimension, row identity and dataset checksum are verified. No SPEARMINT/MINT embeddings or NetMHCstabpan predictions are used. Optional zero-shot, second-model and groove-domain arms have not been run.

Raw full-budget predictions are in `per_row.csv`. `per_fold.csv` contains scores for **all budgets**, identified by `fraction`; filter `fraction == 1` for the full-budget table. Per-fit raw predictions and chosen hyperparameters remain in `jobs/`. Compute metadata include GPU wall time, memory, throughput and CPU fit time; the plot's GPU axis is not total financial cost. v2 reused existing embeddings and required no additional GPU extraction.

## Demo

`python app.py` serves <http://127.0.0.1:7860>. Two examples cover a well-measured and a sparsely measured allele **in this dataset**. Cards show each arm, the half-life/tier, available training-fold counts, nearest better-measured neighbour, reported zero fraction and published SPEARMINT split membership. Training files are distinguished from validation/test files.

Known pairs display their excluded-peptide-fold prediction. The range across all five CV fits is explicitly a sensitivity diagnostic, not calibrated uncertainty; four fits may contain the measured pair. Novel peptides load ESM-2 locally on first use and can take several minutes. Negative log-scale predictions are clipped to zero only for display.

## Limitations and validation

Zeros may be left-censored but are treated as exact. C67S constructs are excluded. Only 9-mers/class I, one common Ridge/PCA specification and one pLM are covered. Main allele evaluation permits cross-allele peptide sharing; near-sequence similarity is not clustered. The 34-contact-residue pseudosequence and synthetic linker can be out of distribution. Per the source description, labels average at least two experiments, but per-replicate variance is unavailable: the measurement-noise ceiling is unestimated. No numerical ceiling, population representation, laboratory savings or clinical benefit is claimed.

Tests cover grouping, strict/locus boundaries, zero handling, eligible/absent allele auditing, equal macro weighting, pooled versus macro behaviour, paired-CI alignment, training-only preprocessing, cache validation and the corrected IEDB audit. Live examples and clean-clone checks are recorded under `results/benchmark/`. Browser visual inspection was unavailable; static scientific figures were inspected directly.

The old IEDB active-learning study remains under `scripts/` and its original result directories; the final test remains untouched. It showed pure diversity acquisition losing to random. Feature geometry is a possible explanation, not a demonstrated cause. Uncertainty acquisition was never tested, so this is not evidence that active learning generally fails.

Method references: [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html), [paired t intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html). See [implementation status](reports/implementation_status.md) for scope.
