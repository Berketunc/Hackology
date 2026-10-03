# Benchmark findings: when does pretraining help?

## P1: data-rich held-out peptides should show little or no pLM advantage

Observed: the BLOSUM neural baseline reached Spearman **0.753 ± 0.007**, compared with **0.523 ± 0.008** for ESM-2 mean pooling and **0.406 ± 0.024** for conditioned peptide residues. Both ESM-2 arms lost to the neural baseline in every fold. With the head matched, BLOSUM Ridge also beat both ESM-2 arms (0.556 ± 0.010).

**Assessment: supported for these frozen representations, this head and this dataset.** This does not establish that pretraining has no value in all data-rich problems.

## P2: held-out alleles should reveal an advantage

Observed: the neural baseline reached **0.473 ± 0.151**; mean-pooled ESM-2 reached **0.292 ± 0.142** and conditioned residues **0.268 ± 0.109**. Both ESM-2 arms lost to the neural baseline in every held-out-allele fold. The allele-ID Ridge control reached 0.210 ± 0.100, despite having no learned identity coefficient for unseen alleles; it can still use generic peptide sequence signals.

There is a limited matched-head result: ESM-2 mean exceeds BLOSUM Ridge's pooled fold mean (0.292 vs 0.246), but wins only **2 of 5 folds**. Conditioned residues also win only 2 of 5. More importantly, average within-allele Spearman is **0.385 for the neural baseline, 0.232 for BLOSUM Ridge, 0.165 for ESM-2 mean, and 0.188 for conditioned residues**. The apparent pooled Ridge advantage does not translate into better peptide ranking within an allele.

The identity-stratified plot shows the neural baseline ahead of both pLM arms in every identity band's average within-allele correlation. The joint distance/support figure has a few positive pLM cells, but **every positive cell contains only one allele**. These are leads for further work, not reliable evidence of a favourable population.

**Assessment: not supported against the challenge's supervised neural baseline.** Comparing only against an allele-identity control would overstate the result.

Every held-out allele has exactly **zero** same-allele training measurements. That quantity cannot be usefully binned. The tables retain it, and the support panels separately use the number of measurements for the nearest training allele. They do not relabel those measurements as belonging to the held-out allele.

## P3: the pLM advantage should shrink with training-set size

Observed fold-mean Spearman gaps relative to the BLOSUM neural baseline:

| Regime / representation | 10% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|
| Peptide / ESM-2 mean | -0.076 | -0.146 | -0.197 | -0.230 |
| Peptide / ESM-2 residue | -0.206 | -0.275 | -0.321 | -0.347 |
| Allele / ESM-2 mean | -0.084 | -0.092 | -0.127 | -0.181 |
| Allele / ESM-2 residue | -0.080 | -0.100 | -0.147 | -0.205 |

**Assessment: the fold-mean gap decreases monotonically on this grid, but it is already negative at 10%.** The neural baseline improves more with data. There is no demonstrated low-data pLM benefit in the tested range. These are nested row subsets, one seeded subset per outer fold; monotonicity of these averages is not a statistical law or a claim about each fold.

## P4: extraction should matter more than model choice

Observed: with the same ESM-2 checkpoint and Ridge-head rule, mean pooling beats conditioned-residue extraction on the primary pooled metric by 0.117 for peptides and 0.024 for alleles. Conditioned residues instead improve within-allele correlation over mean pooling, although both remain below the sequence baselines. Thus the representation and aggregation metric materially change the conclusion.

**Assessment: extraction matters, but the full prediction is untested.** A second pLM was not run, so we cannot compare extraction effects with model-choice effects. The expected superiority of the conditioned-residue arm did not appear on the primary metric. PCA retains at most 256 directions for every Ridge arm; the result is conditional on that compression and the synthetic linker. No post-result representation tuning was performed.

## Does H1 survive?

H1 does **not** receive evidence of a reliable practical pLM advantage in this experiment. Its data-size prediction has the expected direction, but neither tested representation beats the supervised neural baseline at any of the four fold-mean training budgets in either regime. Greater distance does not establish a positive advantage: the sparsest favourable joint cells are single-allele observations. The broader possibility of helpful pretraining remains open because only two frozen extractions of one pLM and one common Ridge-head specification were tested.

A defensible headline is: **on 27,031 organiser-provided peptide–HLA pairs, a small sequence-trained neural network beats both tested frozen ESM-2 representations, including held-out alleles; pooled metrics alone can suggest gains that disappear when evaluating peptide ranking within alleles.**

## Results and compute

| Arm | Held-out peptide ρ | Held-out allele ρ |
|---|---:|---:|
| BLOSUM MLP | 0.753 ± 0.007 | 0.473 ± 0.151 |
| BLOSUM Ridge | 0.556 ± 0.010 | 0.246 ± 0.145 |
| ESM-2 residue | 0.406 ± 0.024 | 0.268 ± 0.109 |
| ESM-2 mean | 0.523 ± 0.008 | 0.292 ± 0.142 |
| Allele-ID Ridge | 0.559 ± 0.010 | 0.210 ± 0.100 |

Values are mean ± sample SD across five folds, not confidence intervals. Raw predictions, paired fold gaps, macro-allele scores, RMSE, Pearson and tier AUC are in `results/benchmark/`.

| Representation | GPU function seconds | Local extraction wall seconds | Peak allocated GPU memory | CPU fit/predict wall seconds, 10 full-budget folds |
|---|---:|---:|---:|---:|
| BLOSUM MLP | 0 | — | — | 25.72 |
| BLOSUM Ridge | 0 | — | — | 7.77 |
| Allele-ID Ridge | 0 | — | — | 38.33 |
| ESM-2 mean | 23.40 | 57.29 | 1.43 GB | 24.31 |
| ESM-2 residue | 83.88 | 252.44 | 1.48 GB | 195.58 |

Extraction used an NVIDIA L4, float16 model inference and float32 stored features. GPU function seconds include model loading inside the worker but exclude image/container startup and transfer. Local wall time includes waiting and transfer. These are measured timings, not billed GPU time or dollar estimates. CPU numbers include preprocessing, fitting and prediction and depend on the local machine and concurrent load; the primary compute figure shows GPU extraction only. Feature extraction is paid once and reused across folds and learning-curve budgets.

## Provenance, scope and limitations

The organiser snapshot contains 28,166 pairs, 75 allele labels and 5,679 recorded zeros. Excluding 1,135 engineered C67S rows leaves 27,031 pairs, 72 alleles and 4,711 zeros. Lengths, finite nonnegative labels, pair uniqueness, per-allele sequence consistency and input checksums are asserted. Splits are saved before fitting. Every learned transform is fitted only on outer training rows; MLP early-stopping validation is grouped inside those rows. The same Ridge pipeline and alpha grid are used across all Ridge representations. The MLP is a separate, stronger nonlinear comparator.

Ridge's internal LOO alpha selection uses preprocessing fitted on the outer training fold rather than refitting transforms per LOO sample; its tuning estimate is not an independent validation score. Outer test rows are excluded. Seed 0 determines outer splits; outer fold numbers determine model, PCA, validation and nested-subset seeds. `design_clarifications.json` records that existing source-code schedule explicitly.

Zeros may be left-censored and are treated as exact recorded zeros. Only class-I 9-mers are covered; engineered alleles are excluded. Peptide grouping is exact, not similarity-clustered. The allele regime can share peptide sequences between different alleles. Small alleles give noisy correlations. Fold variation is large and folds share training rows. Different populations of peptides/alleles and later method selection need new evaluation.

Pseudosequences contain non-contiguous contact residues and can be out of distribution for a protein model. `peptide + GGGG + pseudosequence` is a synthetic input, not a validated structural complex. Frozen ESM-2 was used; no MINT, SPEARMINT or NetMHCstabpan predictor was used in this benchmark. General ESM-2 pretraining sequence exposure is unknown. SPEARMINT pair overlap is retained for demo transparency, not used as a model input.

Optional zero-shot, second-pLM, groove-domain, structure, and contamination-case-study arms were deferred in favour of the minimum viable submission, learning curves and reproducibility. Therefore P4's model-choice contrast and the groove-domain mitigation remain open. The Gradio demo's fold range is a sensitivity diagnostic, not calibrated predictive uncertainty; for known pairs, four folds may have trained on the pair, while the central displayed prediction is from its excluded-peptide fold. The local server and both example API calls were tested; visual browser inspection was unavailable in this session.

The previous IEDB active-learning experiment remains separate, and its final-test labels have not been evaluated. Its negative diversity result is preserved. Geometry is a possible mechanism, not an experimentally established cause. No laboratory savings, therapeutic improvement or clinical utility is claimed.
