# v2 findings: representation benefit depends on the evaluation regime

The primary endpoint is now **macro within-allele Spearman**: calculate a correlation for each held-out allele, then give eligible alleles equal weight. Eligibility requires at least 20 test rows and 10 distinct positive recorded half-lives. Main tables report the mean and SD of these per-fold macros. Pooled Spearman is secondary. The sequence-aware BLOSUM MLP is the reference; allele-ID Ridge is only an illustrative floor.

## P1 — data-rich held-out peptides should show no pLM advantage

**Prediction:** frozen protein representations offer little or no advantage over the sequence-aware supervised baseline when labels are plentiful and peptide groups are held out.

**Observed:** the MLP scores **0.612 ± 0.013**, ESM-2 independent means **0.168 ± 0.021**, and ESM-2 joint sequence encoding **0.215 ± 0.029**. Relative to the MLP, the mean arm's paired difference is **−0.443 [95% CI −0.471, −0.416]**; the joint arm's is **−0.396 [−0.424, −0.369]**.

**Assessment:** consistent with P1 for these frozen representations and this fitting procedure. No conclusion about all protein models or all downstream heads follows.

## P2 — greater allele distance should expose an advantage over the sequence-aware baseline

**Prediction:** a pLM should beat the BLOSUM-pseudosequence MLP on unseen alleles, with a larger benefit farther from the nearest training allele.

**Observed on ordinary allele holdout:** MLP **0.392 ± 0.126**, independent means **0.160 ± 0.067**, joint encoding **0.190 ± 0.081**. Paired differences versus the MLP are **−0.233 [−0.322, −0.143]** and **−0.202 [−0.274, −0.131]**, respectively. Both remain worse than the sequence-aware reference. Beating allele-ID Ridge would not establish transfer to an unseen sequence and is not the hypothesis test.

**Distance result:** the disadvantage gets smaller as the nearest training allele becomes less similar. The headline figure includes both locus stress tests. The dataset has HLA-A and HLA-B only; no HLA-C measurements are available, so the tests train B/test A and train A/test B.

| Stress test | MLP macro | ESM-2 mean macro | Mean − MLP [95% CI] | Joint macro | Joint − MLP [95% CI] |
|---|---:|---:|---:|---:|---:|
| Hold out A | 0.054 | 0.073 | +0.019 [−0.053, +0.086] | 0.125 | +0.071 [+0.004, +0.133] |
| Hold out B | 0.070 | 0.069 | −0.002 [−0.058, +0.057] | 0.098 | +0.028 [−0.029, +0.086] |

These per-locus intervals bootstrap paired allele scores (34 eligible alleles each), conditional on the fixed training/test split. They are not intervals from repeated independent locus experiments. Across only two locus folds, joint encoding's mean difference is **+0.050 [two-fold t CI −0.226, +0.325]**, which is highly imprecise. The A-holdout result is an exploratory signal at the nominal interval level; no adjustment for the many comparisons is applied.

**Assessment:** no advantage in the primary allele regime, but the distance pattern and A-locus result are consistent with a possible benefit under greater shift. This is more nuanced than either “pretraining wins” or “pretraining never helps.” Scores under locus transfer remain low in absolute terms.

All held-out alleles have **zero same-allele training rows**. `distance_stratified.csv` reports that actual zero-count stratum. The additional support panel uses measurements of the **nearest training allele**, explicitly a different variable. These are measurements in this dataset, not human population frequencies.

## P3 — the pLM advantage should shrink with label budget

**Prediction:** the arm-minus-MLP difference decreases as more labels become available.

**Observed:** fold-mean differences decrease monotonically across the 10/25/50/100% grid for both pLM arms in both primary regimes. They are already negative at 10%. For allele holdout at 10%, the mean-arm difference is **−0.057 [−0.121, +0.007]** and the joint-arm difference **−0.028 [−0.087, +0.030]**: those intervals include zero, not evidence of a demonstrated benefit. At 100%, the corresponding differences are **−0.233 [−0.322, −0.143]** and **−0.202 [−0.274, −0.131]**.

**Assessment:** the direction predicted by P3 appears in the point estimates, but no positive average advantage is demonstrated on this budget range. This is one nested seeded subset schedule per fold, not a guarantee of monotonicity for other draws. Every plotted gap has a paired five-fold t interval; all values are in `paired_comparisons.csv`.

## P4 — extraction matters more than which pLM is chosen

**Prediction:** representation extraction affects results more than model identity.

**Observed:** using the same ESM-2 checkpoint and Ridge procedure, joint encoding improves macro Spearman over independent mean pooling by **+0.047 [+0.027, +0.067]** on peptide holdout and **+0.031 [−0.002, +0.063]** on allele holdout. This reverses the ordering under pooled Spearman and illustrates why the metric change matters. The allele interval includes zero.

**Assessment:** the extraction comparison is measured; the “more than model choice” part remains untested because optional Arm D was not run. “Joint sequence encoding” describes the synthetic input `peptide + GGGG + pseudosequence`; it does not assert physical peptide–HLA interactions or a structural complex representation.

## Strict peptide-sharing control

Permissive allele fold 0 trains on 22,879 rows. Removing every training pair whose peptide occurs in the held-out alleles removes **15,085 rows**, leaving **7,794**, while preserving the same **4,152 test rows** and **12 macro-eligible alleles**. No peptide or allele overlaps remain between strict training and test.

| Arm | Permissive macro | Strict macro | Strict − permissive [95% paired-allele CI] |
|---|---:|---:|---:|
| BLOSUM MLP | 0.264 | 0.147 | −0.117 [−0.251, +0.009] |
| ESM-2 mean | 0.069 | −0.024 | −0.093 [−0.160, −0.036] |
| ESM-2 joint | 0.093 | 0.009 | −0.084 [−0.170, −0.001] |

The joint-minus-MLP difference in the strict setting is **−0.138 [−0.256, −0.005]**; the mean-minus-MLP difference is **−0.170 [−0.273, −0.061]**. Intervals condition on this one fold. Removing peptide sharing also cuts the label budget substantially, so the reduction is **not an isolated causal estimate** of peptide sharing. No matched-size causal control is claimed.

## Does H1 survive?

The label-budget and distance patterns are consistent with the idea that pretraining's inductive bias matters more under scarcity or shift. A modest joint-encoding signal appears when HLA-A is entirely held out. However, the pLM arms lose to the sequence-aware reference in the main allele regime, and only two loci are available. **H1 remains plausible but is not established as a reliable general benefit.** The information-processing argument motivates this hypothesis; it does not mathematically guarantee the observed distance or budget trends.

## Full comparison

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

`pooled − macro` is a descriptive aggregation gap. It can reflect between-allele level shifts and unequal sample weighting, but it is not a mathematical decomposition or causal estimate of offset contribution. A negative gap is possible, as the locus results demonstrate.

## Zero-heavy alleles: lead with tier AUC

The main dataset retains all 4,711 recorded zeros. At the prespecified ≥50% zero threshold, four alleles require particular caution because of tied labels:

| Allele | Rows | Zeros | MLP AUC ≥2 h | MLP AUC ≥6 h | Joint AUC ≥2 h | Joint AUC ≥6 h |
|---|---:|---:|---:|---:|---:|---:|
| HLA-B*41:01 | 368 | 59.8% | 0.852 | 0.765 | 0.763 | 0.790 |
| HLA-B*45:01 | 368 | 67.1% | 0.774 | 0.736 | 0.668 | 0.550 |
| HLA-B*51:01 | 353 | 50.7% | 0.705 | 0.740 | 0.581 | 0.575 |
| HLA-B*55:01 | 378 | 55.6% | 0.934 | 0.964 | 0.758 | 0.731 |

These AUCs use each allele's held-out-allele predictions. All arms and budgets are available in `high_zero_allele_auc.csv`; `high_zero_summary.csv` contains the full-budget allele regime. High zero fraction alone does not remove an allele from the macro; only the declared eligibility rule does. AUC is undefined if a threshold has only one class, and such values remain missing rather than being filled.

The allele/locus macro excludes HLA-A*68:02 (16 rows), HLA-A*69:01 (15 rows, 8 distinct positive values), HLA-B*13:02 (7 rows, 4 distinct positive values), and HLA-B*40:02 (19 rows). Peptide-fold eligibility is evaluated separately inside each test fold, so additional sparse alleles fail that threshold. Every exclusion, including zero-observation alleles in peptide folds, is in `excluded_alleles.csv`; all included and excluded counts are in `allele_metric_audit.csv`.

## Compute and selection procedure

The original frozen ESM-2 features were reused; **no new GPU extraction was needed for v2**. Their measured one-time extraction costs remain 23.40 GPU-function seconds for independent means and 83.88 seconds for joint encoding, with peak allocated GPU memory 1.43 and 1.48 GB. Inference throughput was about 623 individual sequences/s for the mean arm (5,705 unique peptide/HLA inputs) and 338 joint sequences/s for the joint arm (27,031 pairs). The units and deduplication differ; these are not identical throughput workloads. `compute.csv` also reports GPU-function throughput, CPU fit/predict wall time and incremental v2 CPU work. GPU times include worker model load but not image/container startup, transfer or a full billing total.

All Ridge arms use the same StandardScaler → randomized PCA (up to 256) → RidgeCV architecture and selection procedure. Alpha is selected **independently for each arm/fold/budget** from 0.01 through 100,000 using training-only LOO, not fixed to a shared value. Full-budget selections are 1,000 for mean ESM-2, 10,000 for joint ESM-2, and 0.01–100 for the sequence Ridge controls. Identical selected values across an arm's folds are an outcome of tuning, not a fixed setting. PCA/scaling exclude outer test rows; their transforms are not refitted inside each internal LOO alpha-selection step. The MLP is the separate nonlinear reference, with training-only grouped early stopping. Per-fit parameters, seeds, training-row hashes and reuse provenance are in `fit_compute.csv` and `jobs/*.json`.

## Noise ceiling and limitations

Per the source description in the brief, labels average at least two experiments. No per-replicate variance is supplied, so assay reproducibility caps achievable correlation by an **unknown** amount; effect sizes must be read against this unestimated ceiling.

Zeros may be left-censored but are modelled as exact recorded zeros. Engineered C67S constructs are excluded. The study covers 9-mers and class I only. The main allele split deliberately permits peptide sharing across alleles; the strict check covers one fold only. Exact peptide grouping does not remove near-sequence similarity. Only one common Ridge/PCA-head specification and one general pLM were evaluated; optional masked-likelihood, second-pLM, groove-domain, structure and contamination arms remain unrun. Pseudosequences consist of non-contiguous residues, and a synthetic linker can be out of distribution. General pretraining sequence exposure is unknown; no stability-trained checkpoint or NetMHCstabpan comparator is used.

Nominal fold t intervals are descriptive: folds share training rows and there are only five folds (two for locus). Paired-allele bootstrap intervals condition on the observed split and do not model dependence among similar alleles, replicate noise or method selection. Stratified comparisons are exploratory and unadjusted for multiplicity. The v2 metric reanalysis followed inspection of v1 outcomes; it is not a fresh confirmatory experiment. The 15 new locus/strict fits followed the saved v2 design. The 200 original fits and v1 reports remain archived with unchanged predictions.

Dataset counts describe only this measurement collection, not human population representation. No laboratory savings or clinical benefit is claimed. The previous IEDB diversity-acquisition result remains separate, with its final test untouched. Feature geometry is a hypothesized mechanism, not a causal finding; uncertainty-based acquisition was never tested, so the conclusion is “pure diversity lost,” not “active learning does not work.”

Method references: [SciPy paired t confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [training-only RidgeCV alpha selection](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html).
