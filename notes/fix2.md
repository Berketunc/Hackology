# fix.md — implementation brief (v2)

Self-contained brief for a CLI coding agent. You will not have the conversation that
produced this. Everything you need is here. Work top-to-bottom; phases are ordered so
that stopping early still leaves a coherent submission.

**Deadline-bounded.** Assume ~16 working hours. If you are behind, drop from the bottom
(Phase 6 first), never from the middle.

**v2 changes:** primary metric is now macro-averaged *within-allele* Spearman, not
pooled; P2 is stated against the sequence-aware baseline rather than the one-hot
strawman; "complex-conditioned" renamed to "joint encoding"; added locus-level holdout,
peptide-bleed control, per-arm hyperparameter tuning, and paired confidence intervals.

---

## 0. Context

This repo currently contains a retrospective active-learning study on peptide-HLA
stability built on an IEDB-derived subset (5,815 pairs, 10 alleles). That work stays,
but it is no longer the headline. The project is being re-aimed at the actual challenge
question using the dataset the organisers provided.

**Research question.** Under what conditions, if any, do protein foundation model
representations improve peptide-HLA stability prediction over a sequence-aware
supervised model trained on the same sequences — and what do those conditions cost in
compute?

**Central hypothesis (H1).** A protein language model receives the same peptide and HLA
sequence information a supervised model receives. By the data-processing inequality, a
deterministic transform of that sequence cannot create information about the label that
was not already recoverable from it. What pretraining on large-scale protein sequence
corpora supplies is an **inductive bias** — a prior that makes the right function easier
to find from finite samples. Therefore its benefit should scale with the generalization
distance the evaluation demands and with label scarcity, and approach zero when labels
are plentiful and the test distribution matches training.

*(Do not write "UniRef" — ESM-2, ESMC and ProtT5 were pretrained on different corpora.
Say "large-scale protein sequence corpora" unless you verify each checkpoint's corpus.)*

**Predictions.**

| # | Prediction |
| --- | --- |
| P1 | Held-out *peptide*, data-rich: no advantage over the sequence-aware baseline, possibly a penalty. (Already observed on IEDB — this is the control.) |
| P2 | Held-out *allele*: advantage appears **over the BLOSUM-pseudosequence baseline**, which receives identical sequence information. Advantage grows with distance from the nearest training allele. |
| P3 | Advantage shrinks monotonically with training-label budget. |
| P4 | How the representation is extracted matters more than which pLM is chosen. |

P2 is deliberately *not* stated against a one-hot allele indicator. That baseline cannot
represent an unseen allele at all, so beating it proves nothing. Include it as an
illustrative floor, never as the comparator.

---

## 1. Hard constraints — read before writing code

**DO NOT:**

1. **Do not benchmark against NetMHCstabpan.** The brief states it is trained on the
   entirety of this dataset and is an unfair comparator.
2. **Do not use a SPEARMINT or MINT checkpoint for embeddings.** Trained on this data.
   Only general pretrained models (ESM-2, ESMC, ProtT5).
3. **Do not use the 365-aa full-length HLA heavy chain.** Use `hla_pseudoseq` (34
   contact residues) as primary and `hla_seq` (182-aa alpha1/alpha2 groove) as an
   ablation. Measured on mean-pooled ESM-2, 10 full chains had pairwise cosine
   0.9977–0.9999, and after standardization the allele consumed 50% of feature variance.
4. **Do not mean-pool as the only extraction method.** Mean-pooled 9-mer ESM-2
   embeddings here are highly anisotropic: mean pairwise cosine 0.969, ~22 effective
   dimensions of 1280.
5. **Do not fit any preprocessing on test rows.** A prior version of this repo fit a
   `DictVectorizer` on all partitions while its docstring claimed pool-only. Fit
   vectorizers, scalers and PCA on **training rows only, inside each fold**.
6. **Do not use one fixed hyperparameter across arms.** A prior version used
   `Ridge(alpha=10)` for both a ~275-dim one-hot and a 2560-dim embedding feature set
   and concluded the embedding was worse; most of that gap was a regularisation scaling
   artifact. **Same head architecture and same selection procedure for every arm, with
   the hyperparameter chosen per-arm on training folds only.**
7. **Do not fit any model before splits are written to disk.**
8. **Do not drop the zero-valued rows.** See §3.
9. **Do not claim anything about human population representation** from dataset row
   counts. Row counts measure representation *in this dataset*. See §3.

**ALWAYS:**

- Write every result to CSV under `results/` and commit it. The current `.gitignore`
  contains `*.csv`, which silently excluded every dataset and every results table.
- Record and use explicit seeds. Write the design to `design.json` **before** running.
- Assert the embedding checkpoint name and dimension against
  `data/processed/embeddings_meta.json` before using cached embeddings.

---

## Phase 0 — unblock the repo (~45 min, do first)

- [ ] **0.1** Replace `.gitignore`. Keep ignoring `.env`, `__pycache__/`, `.DS_Store`.
      **Stop ignoring** `results/**/*.csv` and `data/processed/*.csv`.
      *Accept:* `git status` shows previously-invisible results CSVs as untracked.
- [ ] **0.2** Delete committed `.DS_Store` and `scripts/__pycache__/`.
- [ ] **0.3** Create `src/config.py` with `ROOT = Path(__file__).resolve().parents[1]`
      and derived `DATA`, `RESULTS`, `REPORTS`. Replace the 11 hardcoded
      `/Users/berketunc/...` paths across the 8 files in `scripts/`.
      *Accept:* `grep -rn "/Users/" --include=*.py .` returns nothing.
- [ ] **0.4** `requirements.txt`, pinning at least `numpy>=2.0` (code uses
      `np.trapezoid`), `pandas`, `scipy`, `scikit-learn`, `torch`, `transformers`,
      `biopython`, `gradio`, `matplotlib`.
- [ ] **0.5** Delete `scripts/embed_esm2.py` — it targets `esm2_t30_150M_UR50D`
      (640-dim) while `scripts/modal_embed.py` targets `esm2_t33_650M_UR50D` (1280-dim),
      and both write the same `.npz` with no dimension check. Keep the Modal one.

---

## Phase 1 — data layer and splits (~2.5 h, blocks everything)

Dataset: the organiser-provided CSV. Columns exactly: `allele`, `peptide`,
`thalf_hours`, `hla_seq`, `hla_pseudoseq`. Expect 28,166 rows, all peptides 9-mers, 75
allele labels, `hla_seq` 182 residues, `hla_pseudoseq` 34 residues. **Verify with
assertions; do not trust this brief.**

- [ ] **1.1** `src/data.py::load_rasmussen() -> pd.DataFrame`
      - Assert all `len(peptide) == 9`, all `len(hla_pseudoseq) == 34`, no nulls.
      - Add `y = np.log1p(thalf_hours)`.
      - Add `is_engineered` = allele label contains `(` — the three `(C67S)` constructs,
        ~1,135 rows. **Exclude from the main analysis**, record the exact count.
      - Add `tier` ∈ {`low`,`intermediate`,`high`} at `<2 h`, `2–6 h`, `≥6 h`.
      - Add `locus` = the letter after `HLA-` (A/B/C).
      *Accept:* `data/manifests/rasmussen_manifest.json` with row/allele/peptide counts,
      zero count, zero count **per allele**, engineered count, and the file sha256.

- [ ] **1.2** `src/splits.py` — three split regimes, all written to `results/splits/`:
      - `make_peptide_folds(df, k=5, seed=0)` — `GroupKFold` on `peptide`. **Control.**
      - `make_allele_folds(df, k=5, seed=0)` — group on `allele`, every allele held out
        exactly once. **Main test.**
      - `make_locus_holdout(df)` — hold out an entire locus (e.g. train A+C, test B).
        **Maximum-distance stress test.** Cheap, and it gives the generalization-distance
        axis real dynamic range if the allele folds turn out to be easy.
      - Assert zero group overlap between train and test in every fold.
      *Accept:* one CSV per regime (`row_index, fold`) plus `results/splits/design.json`
      recording k, seeds, exclusions, written **before** any model is fit.

- [ ] **1.3** `src/splits.py::nearest_train_allele_identity(df, folds)` — for each
      held-out allele, max pseudosequence identity (matches/34) to any training allele.
      *Accept:* `results/splits/allele_distance.csv` with
      `allele, fold, n_train_rows, n_test_rows, zero_fraction, max_identity_to_train`.

- [ ] **1.4 Peptide bleed control.** Under the allele split, a test-allele peptide may
      also appear in training paired with a *different* allele. Treat the permissive
      version as primary (it is realistic — you do know about other alleles), but add
      a `strict` variant for **one fold only** that additionally removes every training
      row whose peptide appears in the held-out allele. Report both.
      *Accept:* `results/splits/allele_folds_strict_fold0.csv` and a row count delta.

---

## Phase 2 — baseline and metrics (~2.5 h)

- [ ] **2.1** `src/features.py`
      - `blosum_encode(seq)` using BLOSUM62 from `Bio.Align.substitution_matrices`.
        Peptide → 9×20; pseudoseq → 34×20; flatten and concatenate.
      - `onehot_allele(df)` — illustrative floor only, never the comparator (see §0).
- [ ] **2.2** `src/models.py`
      - `fit_baseline_nn(X, y)` — small MLP (≈256, ≈64, ReLU), early stopping on an
        inner validation split carved from **training rows only**. This is the brief's
        own suggested baseline: *"a simple supervised neural network trained on peptide
        and HLA pairs"*. **This is the reference line, not the one-hot model.**
      - `fit_ridge(X, y)` — `RidgeCV` over a wide alpha grid, chosen on training rows
        only. Used as the fixed head across representation arms.
- [ ] **2.3** `src/evaluate.py` — **metric definitions matter more than anything else in
      this file; get them right.**
      - `macro_within_allele_spearman(per_row)` — **PRIMARY METRIC.** Compute Spearman ρ
        *within each held-out allele*, then average across alleles with equal weight.
        Rationale: the operational task is ranking candidate peptides for one patient's
        allele. A pooled Spearman across alleles rewards getting between-allele level
        shifts right even if within-allele ranking is random, and is dominated by the
        largest alleles.
      - Minimum sample rule: require ≥20 test rows and ≥10 distinct non-zero half-lives
        for an allele to enter the macro average. Report excluded alleles separately
        with their counts — never drop silently.
      - `pooled_spearman(per_row)` — **secondary diagnostic.** Also report
        `pooled − macro`: the gap quantifies how much of the pooled score comes from
        allele-level offsets rather than within-allele ranking.
      - `rmse_log1p`, `pearson`.
      - `tier_auc(y_hours, y_pred, threshold)` at 2.0 and 6.0. Robust to the zero
        inflation; lead with this for alleles whose zero fraction is high.
      - `paired_delta_ci(arm, baseline, by="fold")` — per-fold paired difference with a
        bootstrap or t-based 95% CI. **Every headline comparison reports a CI, not just
        two means.** A prior version of this repo reported "0/10 seeds better" with no
        interval; do not repeat that.
      - `stratify_by_allele_distance(per_row, allele_distance)` — metric per
        identity band and per `n_train_rows` band.

---

## Phase 3 — representation arms (~4 h, the core)

All arms use the **same splits, same folds, same metric code, same head architecture and
same hyperparameter-selection procedure**, with the hyperparameter tuned per-arm on
training rows. Only the representation varies.

- [ ] **3.1 Arm A — mean-pooled ESM-2.** `concat(mean_pool(peptide), mean_pool(pseudoseq))`,
      checkpoint `facebook/esm2_t33_650M_UR50D`. Expected to underperform the baseline;
      that is P1.
- [ ] **3.2 Arm B — joint peptide–HLA encoding.** Tokenize `peptide + linker + pseudoseq`
      as **one sequence** (short glycine linker, e.g. `GGGG`; document the choice). Take
      hidden states at the **9 peptide positions** → 9×1280. PCA fit on training rows
      only if the head is slow.
      **Naming discipline:** call this *joint sequence encoding*, not "complex-conditioned"
      or "interaction-aware". ESM-2 is trained on single chains; cross-token attention
      after concatenation does not establish that the model represents a physical
      peptide–HLA complex. Whether joint beats independent encoding is the empirical
      question (P4) — do not assert it in the name.
      **Prioritise this arm if time is short.** It is the cleanest test of P4 and the most
      likely to move the number.
- [ ] **3.3 Arm C — zero-shot masked log-likelihood.** No training. Mask each of the 9
      peptide positions in turn within the joint sequence, sum `log p(true residue)`.
      **Subset to ~3,000 stratified pairs** — the brief explicitly sanctions subsetting.
      *Confound to control:* peptide likelihood is dominated by amino-acid composition
      and general proteome statistics, and HLA ligands are a biased sample of proteome
      9-mers. A pooled correlation may reflect "looks like a presented peptide" rather
      than anything allele-specific. **Report the within-allele correlation, not just the
      pooled one.** Exploratory arm; project success must not depend on it.
- [ ] **3.4 Arm D — second pLM.** ESMC or ProtT5, mean-pooled only. Robustness check.
- [ ] **3.5 Arm E — groove instead of pseudosequence.** Re-run the best arm with
      `hla_seq` (182-aa alpha1/alpha2) in place of `hla_pseudoseq`. The pseudosequence is
      34 non-contiguous residues and may be out of distribution for a pLM; the groove is
      a real contiguous domain. This is a biological input ablation and a documented
      risk-mitigation for P2 — report it as a result, not a nuisance.

Run embeddings on Modal using the existing `scripts/modal_embed.py` pattern. **Record
wall-clock GPU seconds, peak memory and throughput for every arm.**

- [ ] **3.6** Orchestrator `src/run_benchmark.py` writing
      `results/benchmark/per_fold.csv` (`arm, split_regime, fold, metric, value`) and
      `results/benchmark/per_row.csv` (`row_index, allele, arm, split_regime, y_true,
      y_pred`). Everything downstream reads these two files.

---

## Phase 4 — headline analysis (~1.5 h)

- [ ] **4.1** Main table: macro within-allele Spearman × arm × split regime, mean ± sd
      over folds, with paired CI vs the supervised baseline. Report pooled Spearman and
      the `pooled − macro` gap alongside.
- [ ] **4.2** **Headline figure.** Held-out-allele macro Spearman per arm, stratified by
      `max_identity_to_train` band and by `n_train_rows`. H1 predicts the pLM advantage
      is largest where both are smallest. Add the locus-holdout point as the
      maximum-distance anchor. **Make this figure the headline whether or not the overall
      delta is significant** — a null overall delta with a rising delta-vs-distance curve
      is still a result.
- [ ] **4.3** Label-budget curves (P3): subsample training rows at 10/25/50/100%, plot
      the arm-minus-baseline delta against training size, with CIs.
- [ ] **4.4** Compute plot: macro Spearman on y, GPU-seconds (or dollars) on x, one point
      per arm.
- [ ] **4.5** `reports/benchmark_findings.md`: for each prediction, what it predicted,
      what happened, whether H1 survives. **A negative result is explicitly welcomed by
      the brief — report it straight.**
- [ ] **4.6** One line on the **noise ceiling**: labels are averages of ≥2 experiments and
      no per-replicate variance is available, so assay reproducibility caps achievable
      correlation by an unknown amount. State that effect sizes should be read against an
      unestimated ceiling. Do not invent a number.

---

## Phase 5 — demo (~2 h, do not skip)

Worth 20% of the judging rubric; there is currently nothing on screen.

- [ ] **5.1** `app.py`, Gradio. Input: a 9-mer peptide and an allele from a dropdown.
      Output card: predicted half-life with an uncertainty band (fold spread is enough);
      stability tier badge; each arm side by side; **how many training measurements exist
      for that allele** and its nearest better-measured neighbour; **whether the pair
      appears in any known model's training data** (reuse `scripts/overlap_report.py`).
- [ ] **5.2** Pre-load two states: one well-measured allele, one sparsely-measured one.
      The demo moment is switching between them and watching the arms separate.

---

## Phase 6 — stretch, only if Phase 5 is green

- [ ] **6.1** Structure arm: Boltz-2 or Chai-1 on 50–100 complexes; ipTM/PAE vs measured
      half-life.
- [ ] **6.2** SPEARMINT contamination case study: its reported performance beside its
      performance on pairs it has not seen.
- [ ] **6.3** Merge the corrected `audit()` from `agenthandoff.md` Appendix A into
      `check_stability_data.py` — the shipped version reproduces 286 unflagged rows
      instead of 6,101. Add a regression test asserting 6,101 / 5,815 / 6.
- [ ] **6.4** *Only if you want the population claim:* pull real HLA allele-frequency
      data (allelefrequencies.net or a cited publication) and join it. Until then the
      language is "alleles with few measurements **in this dataset**", never "underserved
      populations".

---

## Phase 7 — ship (~1 h, reserve it)

- [ ] **7.1** `README.md`: research question, H1, the four predictions, reproduction
      steps, result table, limitations.
- [ ] **7.2** Confirm every results CSV is committed and the repo clones-and-runs clean.
- [ ] **7.3** Limitations, stated plainly: zeros are likely left-censored and modelled as
      exact; `(C67S)` constructs excluded; 9-mers and class I only; one head
      architecture; permissive peptide bleed in the main allele split; unestimated
      measurement-noise ceiling; retrospective, no laboratory claim.

---

## 3. Decisions already made — do not relitigate

- **Zeros stay in.** ~20% of rows have `thalf_hours == 0`. `log1p(0) = 0` is
  well-defined and Spearman handles ties. Excluding them restricts the task to peptides
  already known to bind — the easier and less useful half — and makes the tier metric
  meaningless. **But record the zero fraction per allele**: an allele that is 60% zeros
  has a near-meaningless within-allele Spearman, so lead with tier-AUC for those and say
  which alleles they are. Note left-censoring in limitations; do not implement a Tobit
  model, there is no time.
- **Primary metric is macro-averaged within-allele Spearman.** Pooled is a secondary
  diagnostic.
- **Primary HLA representation is `hla_pseudoseq`**; `hla_seq` is Arm E.
- **Engineered `(C67S)` alleles excluded** from the main analysis, counted and reported.
- **The reference baseline is the sequence-aware supervised NN**, not the one-hot allele
  model.
- **The old IEDB active-learning work stays** under `scripts/`. It is one slide:
  diversity-based acquisition lost to random in 10/10 seeds, the measured reason is the
  feature-geometry defect in constraint 3. Note that uncertainty-based acquisition — the
  stronger member of that family — was never tested, so the slide says "diversity lost",
  not "active learning doesn't work". Do not delete or refactor this code.

---

## 4. Definition of done

Minimum viable submission, in priority order:

1. Phases 0–2: data loads, three split regimes locked on disk, supervised baseline runs
   with macro within-allele Spearman and paired CIs.
2. Arms A and B on peptide-split and allele-split. This alone tests P1, P2 and P4.
3. Headline stratified figure (4.2) and `reports/benchmark_findings.md`.
4. Gradio demo running locally.
5. README and committed results.

Arms C, D, E and all of Phase 6 are upside. **Do not start Phase 6 until Phase 5 runs.**
