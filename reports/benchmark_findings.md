# Matched-head findings: representation and fitting procedure both matter

**Current finding:** prediction depends strongly on both the representation and the downstream fitting procedure. Adding the same 256/64 MLP procedure improves both ESM-2 representations on held-out peptides, but BLOSUM + MLP still leads in both primary regimes. The earlier ESM-2 + Ridge versus BLOSUM + MLP comparison mixed these two factors; the follow-up narrows that gap without reversing the main ranking.

This benchmark uses the **supplied Rasmussen table**, not the archived IEDB pilot: 28,166 9-mer measurements / 75 allele labels before excluding 1,135 engineered C67S rows; 27,031 pairs / 72 alleles after exclusion, retaining 4,711 zeros. The endpoint is macro within-allele Spearman, with ≥20 test rows and ≥10 distinct positive half-lives per eligible allele. Eligible alleles have equal weight, and zero labels remain in their correlations.

## The representation × fitting-procedure comparison

At full budget, peptide-holdout macro Spearman is **0.612** for BLOSUM + MLP, **0.360** for ESM-2 mean + MLP, and **0.377** for ESM-2 joint + MLP. On allele holdout the corresponding scores are **0.392**, **0.237**, and **0.252**.

Holding the representation fixed, the peptide MLP-minus-Ridge gains are **+0.354 [+0.323, +0.385]** for BLOSUM, **+0.192 [+0.179, +0.204]** for ESM-2 mean, and **+0.162 [+0.124, +0.201]** for ESM-2 joint. With MLP procedures matched, the remaining peptide gaps against BLOSUM are **-0.252 [-0.279, -0.224]** and **-0.234 [-0.251, -0.217]**. These are descriptive paired 95% intervals, not causal shares of an error budget.

Full-budget macro within-allele Spearman; five-fold paired t intervals. These are descriptive contrasts of fitting procedures, not a causal decomposition.

| Representation | Peptide Ridge | Peptide MLP | Allele Ridge | Allele MLP |
|---|---:|---:|---:|---:|
| BLOSUM | 0.258 | 0.612 | 0.228 | 0.392 |
| ESM-2 mean | 0.168 | 0.360 | 0.160 | 0.237 |
| ESM-2 joint | 0.215 | 0.377 | 0.190 | 0.252 |

| Contrast (first minus second) | Peptide Δ [95% CI] | Allele Δ [95% CI] |
|---|---:|---:|
| BLOSUM MLP (reference) − BLOSUM Ridge | +0.354 [+0.323, +0.385] | +0.164 [+0.089, +0.240] |
| ESM-2 mean + MLP − ESM-2 mean + Ridge | +0.192 [+0.179, +0.204] | +0.078 [+0.006, +0.150] |
| ESM-2 joint + MLP − ESM-2 joint + Ridge | +0.162 [+0.124, +0.201] | +0.062 [+0.018, +0.106] |
| ESM-2 mean + Ridge − BLOSUM Ridge | -0.090 [-0.107, -0.072] | -0.068 [-0.096, -0.041] |
| ESM-2 joint + Ridge − BLOSUM Ridge | -0.043 [-0.069, -0.016] | -0.038 [-0.050, -0.026] |
| ESM-2 mean + MLP − BLOSUM MLP (reference) | -0.252 [-0.279, -0.224] | -0.155 [-0.264, -0.046] |
| ESM-2 joint + MLP − BLOSUM MLP (reference) | -0.234 [-0.251, -0.217] | -0.140 [-0.214, -0.066] |
| BLOSUM Ridge − Allele-ID Ridge (floor) | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.000] |
| ESM-2 joint + Ridge − ESM-2 mean + Ridge | +0.047 [+0.027, +0.067] | +0.031 [-0.002, +0.063] |
| ESM-2 joint + MLP − ESM-2 mean + MLP | +0.018 [-0.009, +0.044] | +0.015 [-0.046, +0.075] |


![Representation and head comparison](matched_heads.png)

The follow-up adds 26 full-budget fits (13 per new approach) using unchanged cached ESM-2 inputs and the same MLP function, hidden widths, optimizer, grouped early stopping, and fold seed schedule as BLOSUM. Training-row hashes, test rows, and macro eligibility are checked against the original reference in [acceptance checks](../results/benchmark/mlp_extension_checks.json). No PCA was introduced for ESM-2 MLP and no new GPU extraction was performed.

This controls the MLP fitting procedure, not parameter count: input dimensions differ. Comparing Ridge to MLP also changes preprocessing (PCA is used only by Ridge), loss optimization, and regularization. Consequently, this is a decomposition of observed procedure contrasts, not a causal attribution to neural-network capacity alone. For example, the original ESM-2 joint Ridge minus BLOSUM MLP gap equals the joint-minus-BLOSUM contrast under Ridge minus the BLOSUM MLP-minus-Ridge contrast; the alternative path through ESM-2 MLP gives a different descriptive allocation. Neither path proves causality or statistical equivalence.

### Why a linear HLA block cannot change within-allele ranking

For concatenated independent features, the fitted scaler → PCA → Ridge pipeline remains affine in the original inputs: `prediction(p, a) = w_peptide · x(p) + w_HLA · h(a) + b`. Within one allele, `h(a)` is constant. Its contribution is an offset, which cannot change within-allele Spearman. Thus BLOSUM Ridge, allele-ID Ridge, and independent-mean ESM-2 Ridge cannot express allele-dependent peptide rankings for a fixed fitted model. Their peptide functions differ; they are not all the same predictor.

BLOSUM Ridge minus allele-ID Ridge is **-0.000 [-0.001, +0.001]** on peptide holdout and **-0.000 [-0.001, +0.000]** on allele holdout. Their near agreement is consistent with this structural limitation, but **equality between separately trained models is not a mathematical identity**: joint PCA fitting, regularization selection, and the HLA feature block can change the learned peptide coefficients.

Joint ESM-2 features depend on the peptide and allele together before the linear head; they can express context-dependent rankings. MLP heads can also learn interactions from independently concatenated features. This explains a representational capability, not proof that learned attention represents a physical complex or that the capability necessarily improves transfer.

## H1 and P1–P4

H1 proposes that general protein pretraining supplies an inductive bias that helps under scarcity or distribution shift. A deterministic encoding with fixed pretrained weights does not add label information beyond its input sequences; this motivates a finite-sample question rather than proving a distance or budget trend.

### P1 — data-rich held-out peptides

No tested ESM-2 approach beats BLOSUM + MLP. This remains true with the MLP procedure matched, although the original mixed-head deficit overstates the remaining matched-MLP gap. P1 is consistent with the observed full-budget results for these procedures; no claim about all protein models follows.

### P2 — unseen alleles and distance

Under matched MLP procedures, mean ESM-2 minus BLOSUM is **-0.155 [-0.264, -0.046]**, and joint ESM-2 minus BLOSUM is **-0.140 [-0.214, -0.066]**. There is no demonstrated advantage in the primary allele regime. Distance/support analyses remain exploratory. Every held-out allele has zero same-allele training rows; nearest-training-allele measurements are a separate support variable, not population coverage. Whole-locus transfer is retained only in [the appendix](locus_appendix.md).

## P3 — the pLM advantage should shrink with label budget

**Prediction:** the arm-minus-MLP difference decreases as more labels become available.

**Observed for the original Ridge-headed ESM-2 approaches:** fold-mean differences decrease monotonically across the 10/25/50/100% grid for both pLM arms in both primary regimes. They are already negative at 10%. For allele holdout at 10%, the mean-arm difference is **−0.057 [−0.121, +0.007]** and the joint-arm difference **−0.028 [−0.087, +0.030]**: those intervals include zero, not evidence of a demonstrated benefit. At 100%, the corresponding differences are **−0.233 [−0.322, −0.143]** and **−0.202 [−0.274, −0.131]**.

**Assessment:** the direction predicted by P3 appears in the point estimates, but no positive average advantage is demonstrated on this budget range. This is one nested seeded subset schedule per fold, not a guarantee of monotonicity for other draws. Every plotted gap has a paired five-fold t interval; all values are in `paired_comparisons.csv`.


The new ESM-2 + MLP approaches were run at full budget only. P3 has not been tested with matched MLP heads across label budgets.


### P4 — extraction versus model choice

At the Ridge procedure, joint-minus-mean is **+0.047 [+0.027, +0.067]** for peptide holdout and **+0.031 [-0.002, +0.063]** for allele holdout. At the MLP procedure it is **+0.018 [-0.009, +0.044]** and **+0.015 [-0.046, +0.075]**, respectively. Extraction effects depend on the downstream procedure. A second pLM was not run, so the “more than model choice” comparison remains untested.

**Assessment of H1:** a general pretraining advantage is not established. The new control supports a narrower conclusion: fitting procedure explains part of the original deficit, while a substantial disadvantage remains under the tested MLP procedure. Small-budget matched-MLP benefits remain untested.

## Strict peptide-sharing control

Permissive allele fold 0 trains on 22,879 rows. Removing every training pair whose peptide occurs in the held-out alleles removes **15,085 rows**, leaving **7,794**, while preserving the same **4,152 test rows** and **12 macro-eligible alleles**. No peptide or allele overlaps remain between strict training and test.

| Arm | Permissive macro | Strict macro | Strict − permissive [95% paired-allele CI] |
|---|---:|---:|---:|
| BLOSUM MLP | 0.264 | 0.147 | −0.117 [−0.251, +0.009] |
| ESM-2 mean | 0.069 | −0.024 | −0.093 [−0.160, −0.036] |
| ESM-2 joint | 0.093 | 0.009 | −0.084 [−0.170, −0.001] |

The joint-minus-MLP difference in the strict setting is **−0.138 [−0.256, −0.005]**; the mean-minus-MLP difference is **−0.170 [−0.273, −0.061]**. Intervals condition on this one fold. Removing peptide sharing also cuts the label budget substantially, so the reduction is **not an isolated causal estimate** of peptide sharing. No matched-size causal control is claimed.


The original comparisons above are retained as historical controls. All seven approaches and paired-allele intervals are in [the full table](benchmark_table.md) and `results/benchmark/strict_vs_permissive.csv`.


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


## Post-hoc robustness: doubling early-stopping patience

The original allele fits selected epoch ≤2 in **3/5 joint ESM-2** folds and **1/5 mean ESM-2** folds. This prompted a single exploratory change: patience **15 → 30**, keeping the 150-epoch limit, allele-grouped inner validation, features, optimizer, widths, and fold seeds unchanged. All three MLP approaches, including BLOSUM, were rerun on five allele folds and the strict fold: **18 fits**, saved separately from the locked benchmark.

| Approach minus BLOSUM MLP | Allele Δ, patience 15 [95% CI] | Allele Δ, patience 30 [95% CI] |
|---|---:|---:|
| ESM-2 mean + MLP | -0.155 [-0.264, -0.046] | -0.158 [-0.264, -0.052] |
| ESM-2 joint + MLP | -0.140 [-0.214, -0.066] | -0.136 [-0.218, -0.054] |

Both ESM-2 MLP approaches remain below BLOSUM MLP, with nominal paired intervals excluding zero. **5/18 fits still select epoch ≤2 at patience 30.** This increase does not eliminate the early-checkpoint pattern; it does not establish that patience or any other single factor caused the original gap. An epoch-1 checkpoint is trained, and the selected checkpoint can remain early after many subsequent epochs were attempted. We stopped at the planned sensitivity: no patience 60 or alternative validation split.

Five-fold intervals remain nominal and exploratory, and this analysis was chosen after inspecting results. Strict-fold comparisons use a paired-allele bootstrap conditional on that split. Peptide folds were not rerun at patience 30, so their sensitivity to patience is **untested**. The main tables and headline retain the original patience-15 procedure.

See [the complete sensitivity report and stopping-epoch appendix](patience_sensitivity.md), including every original full-budget MLP best epoch, strict results, and within-approach changes; [acceptance checks](../results/benchmark/patience_sensitivity/checks.json) verify unchanged protected outputs and matching training rows, test rows, and eligibility.

## Noise ceiling

The supplied table has no per-replicate values and no duplicate allele–peptide rows; `src/data.py` checks uniqueness. A dataset-specific assay-reproducibility ceiling therefore cannot be estimated. Rasmussen's methods specify geometric means of two independent experiments. External assay-repeatability evidence is documented in [the source review](assay_reproducibility.md), but does not establish a numerical ceiling for this benchmark. IEDB pilot conflicts and SPEARMINT membership flags are not replicate measurements of these labels.

## Compute and reproducibility

Mean and joint extraction originally cost 23.40 and 83.88 GPU-function seconds. The MLP follow-up reuses those same caches. `compute.csv` attributes their original extraction cost to each applicable approach and explicitly marks shared extraction; do not add those entries together as if extraction happened twice. Additional extraction cost for this follow-up is zero. Measured CPU fitting costs are reported separately. GPU-function timing excludes startup, transfer, and a full billing total; throughput workloads differ between separate sequences and joint pairs.

The five original approaches have 215 model/fold/budget jobs. The two full-budget MLP extensions add 26, for **241 jobs and 596,715 full-budget prediction rows**. The original jobs, v2 design, and splits are byte-preserved. `mlp_extension_design.json` records this post-v2 follow-up before fitting; it does not replace the original locked design. The v2 metric change followed inspection of v1 outcomes, and the new head comparison was motivated by v2 inspection: neither is fresh independent confirmation.

The Ridge procedure remains training-only StandardScaler → randomized PCA (up to 256) → independently tuned RidgeCV over 0.01–100,000. Scaling/PCA are not refitted within internal LOO alpha selection. The MLP still uses training-only group validation. Per-fit metadata record training-row hashes, seed, null alpha, and integer best epoch for each new fit.

## Limitations

Zeros may be left-censored but are treated as exact. Engineered C67S constructs are excluded. Main allele evaluation allows peptide sharing; strict evaluation is one fold with fewer labels, not an isolated causal estimate. Near-sequence similarity is not clustered. Only one frozen pLM is tested, and its non-contiguous HLA pseudosequence/synthetic linker can be out of distribution. Pretraining exposure is unknown. No SPEARMINT/MINT stability-trained checkpoint or NetMHCstabpan comparator is used; training on this dataset would compromise that comparison.

Five-fold t intervals are nominal and descriptive because folds share training rows. Allele bootstraps condition on a fixed split and do not model related-allele dependence, replicate noise, or method selection. No multiple-comparison adjustment is made. Two locus folds cannot support a precise general transfer claim. Model point predictions have no calibrated uncertainty interval.

Second-pLM, groove-domain, masked-likelihood, structural-prediction, inverse-folding, and broader contamination studies remain unrun. The archived IEDB active-learning pilot remains separate with its final test untouched; uncertainty acquisition was never tested. Dataset coverage is not population representation. No laboratory savings, clinical benefit, or immunogenicity improvement is claimed.

### Input scope and omitted structure models

The supplied `hla_seq` is a 182-residue α1/α2 groove-domain fragment, without the α3 domain or β2-microglobulin. The actual ESM-2 inputs here are even shorter: the supplied **34 contact residues**, embedded separately or after the synthetic peptide linker. No full native HLA chain or complete complex is encoded. This mismatch to natural protein sequences is a plausible contributor to the observed performance, but its effect has not been isolated; embedding the 182-residue fragment remains an unrun comparison.

A full structure-prediction comparison across 28,166 supplied pairs would require additional large-scale inference, structural input preparation, and a validated mapping from structure to dissociation half-life, beyond this study's compute scope. We have not measured a runtime and do not claim a precise GPU-day estimate. Boltz-2's documented affinity module targets small-molecule–protein binding and reports an IC50-like endpoint, rather than peptide–HLA half-life; it is not a drop-in comparator. Even in a simple two-state kinetic model, `t½ = ln(2)/k_off` whereas `K_d = k_off/k_on`, so equilibrium affinity alone does not determine stability. [Official Boltz prediction documentation](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md#properties-affinity).

## Supporting outputs

- [Full seven-approach tables, including strict and locus](benchmark_table.md)
- [Matched-head comparison table](matched_head_table.md) and [machine-readable contrasts](../results/benchmark/matched_head_comparisons.csv)
- [Learning-gap figure](learning_gap.png): new MLP approaches have only a 100% point
- [Compute figure](compute_performance.png)
- [Historical v2 report](archive/v2/benchmark_findings.md)
