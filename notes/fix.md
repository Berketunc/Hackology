# fix.md — implementation brief

Self-contained brief for a CLI coding agent. You will not have the conversation that
produced this. Everything you need is here. Work top-to-bottom; phases are ordered so
that stopping early still leaves a coherent submission.

**Deadline-bounded.** Assume ~16 working hours. If you are behind, drop from the bottom
(Phase 6 first), never from the middle.

---

## 0. Context

This repo currently contains a retrospective active-learning study on peptide-HLA
stability built on an IEDB-derived subset (5,815 pairs, 10 alleles). That work stays,
but it is no longer the headline. The project is being re-aimed at the actual challenge
question using the dataset the organisers provided.

**New research question.** Under what conditions, if any, do protein foundation model
representations improve peptide-HLA stability prediction over a simple supervised model
trained on the same sequences — and what do those conditions cost in compute?

**Central hypothesis (H1).** A protein language model cannot add information a
supervised model does not already have; both see the same sequence. It can only add
*prior structure* learned from pretraining. Therefore its benefit scales with the
generalization distance the evaluation demands, and approaches zero when the test
distribution matches the training distribution.

**Predictions to test.**

| # | Prediction |
| --- | --- |
| P1 | Held-out *peptide*, data-rich: no advantage over the supervised baseline, possibly a penalty. (Already observed on IEDB — this is the control.) |
| P2 | Held-out *allele*: advantage appears. A one-hot allele indicator cannot generalize to an allele it never saw; a pLM representation of the groove can. |
| P3 | Advantage shrinks monotonically with training-set size. |
| P4 | How you extract the representation matters more than which model you pick. |

---

## 1. Hard constraints — read before writing code

**DO NOT:**

1. **Do not benchmark against NetMHCstabpan.** The challenge brief states it is trained
   on the entirety of this dataset and is an unfair comparator. The baseline is a simple
   supervised NN we train ourselves.
2. **Do not use a SPEARMINT or MINT checkpoint to produce embeddings.** It is trained on
   this data. Only general pretrained models (ESM-2, ESMC, ProtT5).
3. **Do not use the 365-aa full-length HLA heavy chain.** Use `hla_pseudoseq`
   (34 contact residues) as primary and `hla_seq` (182-aa alpha1/alpha2 groove) as a
   secondary arm. The full chains are nearly identical across alleles — measured
   pairwise cosine 0.9977–0.9999 on mean-pooled ESM-2 — and after standardization the
   allele consumed 50% of the feature variance.
4. **Do not mean-pool as the only extraction method.** Mean-pooled 9-mer ESM-2
   embeddings in this dataset are highly anisotropic: mean pairwise cosine 0.969,
   ~22 effective dimensions out of 1280. Mean-pooling is one arm, not the method.
5. **Do not fit any preprocessing on test rows.** A prior version of this repo fit a
   `DictVectorizer` on all partitions while its docstring claimed pool-only. Fit
   vectorizers, scalers and PCA on **training rows only**, inside each fold.
6. **Do not fit any model before splits are written to disk.**
7. **Do not drop the zero-valued rows.** See §3.

**ALWAYS:**

- Write every result to CSV under `results/` and commit it. The current `.gitignore`
  contains `*.csv`, which silently excluded every dataset and every results table from
  the repo. Fix that first (Phase 0).
- Record and use explicit seeds. Write the design to `design.json` **before** running.
- Assert the embedding checkpoint name and dimension against
  `data/processed/embeddings_meta.json` before using cached embeddings.

---

## Phase 0 — unblock the repo (~45 min, do first)

- [ ] **0.1** Replace `.gitignore`. Keep ignoring `.env`, `__pycache__/`, `.DS_Store`.
      **Stop ignoring** `results/**/*.csv` and `data/processed/*.csv`. Large raw inputs
      stay out; results and processed derivatives go in.
      *Accept:* `git status` shows previously-invisible results CSVs as untracked.
- [ ] **0.2** Delete committed `.DS_Store` and `scripts/__pycache__/`.
- [ ] **0.3** Create `src/config.py` with a single `ROOT = Path(__file__).resolve().parents[1]`
      and derived `DATA`, `RESULTS`, `REPORTS` paths. Replace the 11 hardcoded
      `/Users/berketunc/...` paths across the 8 files in `scripts/` with imports from it.
      *Accept:* `grep -rn "/Users/" --include=*.py .` returns nothing.
- [ ] **0.4** Write `requirements.txt`. Must pin at least: `numpy>=2.0` (the code uses
      `np.trapezoid`), `pandas`, `scipy`, `scikit-learn`, `torch`, `transformers`,
      `biopython` (for BLOSUM62), `gradio`, `matplotlib`.
- [ ] **0.5** Delete `scripts/embed_esm2.py`. It targets `esm2_t30_150M_UR50D` (640-dim)
      while `scripts/modal_embed.py` targets `esm2_t33_650M_UR50D` (1280-dim), and both
      write the same `data/processed/embeddings_esm2.npz` with no dimension check.
      Keep the Modal one.

---

## Phase 1 — data layer and splits (~2 h, blocks everything)

Dataset: the organiser-provided CSV. Columns are exactly:
`allele`, `peptide`, `thalf_hours`, `hla_seq`, `hla_pseudoseq`.
Expect 28,166 rows, all peptides 9-mers, 75 allele labels, `hla_seq` 182 residues,
`hla_pseudoseq` 34 residues. **Verify these with assertions; do not trust this brief.**

- [ ] **1.1** `src/data.py::load_rasmussen() -> pd.DataFrame`
      - Read with `dtype=str` for identifier columns, coerce `thalf_hours` to float.
      - Assert: all `len(peptide) == 9`; all `len(hla_pseudoseq) == 34`;
        no nulls in the five columns.
      - Add `y = np.log1p(thalf_hours)`.
      - Add boolean `is_engineered` = allele label contains `(`. These are the three
        `(C67S)`-suffixed constructs (~1,135 rows). **Exclude from the main analysis**
        and record the exact count in the manifest.
      - Add `tier` ∈ {`low`, `intermediate`, `high`} at `<2 h`, `2–6 h`, `≥6 h`.
      *Accept:* a `data/manifests/rasmussen_manifest.json` with row count, allele count,
      peptide count, zero count, engineered count, and the file's sha256.

- [ ] **1.2** `src/splits.py` — two split functions, both returning fold assignments
      written to `results/splits/`:
      - `make_peptide_folds(df, k=5, seed=0)` — `GroupKFold` on `peptide`. Control regime.
      - `make_allele_folds(df, k=5, seed=0)` — group on `allele`, so every allele is
        held out exactly once across the 5 folds. Test regime.
      - Assert zero group overlap between train and test in every fold.
      *Accept:* `results/splits/peptide_folds.csv` and `allele_folds.csv`, each
      `row_index, fold`. Plus `results/splits/design.json` recording k, seeds, exclusion
      rules, written before any model is fit.

- [ ] **1.3** `src/splits.py::nearest_train_allele_identity(df, folds)` — for each
      held-out allele, the maximum pseudosequence identity (matches / 34) to any allele
      in that fold's training set. This drives the headline stratification.
      *Accept:* `results/splits/allele_distance.csv` with
      `allele, fold, n_train_rows, max_identity_to_train_allele`.

---

## Phase 2 — baselines (~2 h)

- [ ] **2.1** `src/features.py`
      - `blosum_encode(seq) -> np.ndarray` using BLOSUM62 from
        `Bio.Align.substitution_matrices`. Peptide → 9×20; pseudoseq → 34×20; flatten
        and concatenate.
      - `onehot_allele(df)` — allele identity indicator. **Only valid in the peptide-split
        regime**; under the allele split it is structurally incapable of generalizing,
        which is the point of P2. Include it anyway and let it fail visibly.
- [ ] **2.2** `src/models.py::fit_baseline_nn(X_train, y_train)` — a small MLP
      (2 hidden layers, ~256 and ~64 units, ReLU, early stopping on an inner validation
      split carved from training rows only). This is the challenge brief's own suggested
      baseline: *"a simple supervised neural network trained on peptide and HLA pairs"*.
      Also provide `fit_ridge(X, y)` with alpha chosen by `RidgeCV` on training rows only.
- [ ] **2.3** `src/evaluate.py`
      - `spearman(y_true, y_pred)` — **primary metric**.
      - `rmse_log1p`, `pearson`.
      - `tier_auc(y_true_hours, y_pred, threshold)` for thresholds 2.0 and 6.0.
      - `stratify_by_allele_distance(per_row_results, allele_distance)` — group held-out
        rows into identity bands and report the metric per band.

---

## Phase 3 — foundation-model arms (~4 h, the core)

All arms use the **same splits, same folds, same metric code, same head**. The only
thing that varies is the representation. Keep the head fixed (`fit_ridge`, or the same
MLP) so a difference is attributable to the representation.

- [ ] **3.1 Arm A — mean-pooled ESM-2.** `concat(mean_pool(peptide), mean_pool(pseudoseq))`,
      checkpoint `facebook/esm2_t33_650M_UR50D`. This reproduces the existing approach on
      the new data. Expected to underperform the baseline — that is P1.
- [ ] **3.2 Arm B — complex-conditioned per-residue.** Tokenize
      `peptide + linker + pseudoseq` as **one sequence** (use a short glycine linker,
      e.g. `GGGG`; document the choice). Take the hidden states at the **9 peptide
      positions** and concatenate → 9×1280. Reduce with PCA fit **on training rows only**
      if the head is slow. Attention conditions the peptide representation on the
      allele — which mean-pooling structurally cannot do. **This is the most likely arm
      to beat the baseline; prioritise it if time is short.**
- [ ] **3.3 Arm C — zero-shot masked log-likelihood.** No training. For each pair, mask
      each of the 9 peptide positions in turn within the complex-conditioned sequence and
      sum `log p(true residue)`. Correlate against `thalf_hours`.
      **Subset to ~3,000 stratified pairs** — 9 forward passes per pair over 28k rows is
      not affordable, and the brief explicitly sanctions subsetting. Record the subset
      size and selection rule.
- [ ] **3.4 Arm D — second pLM.** ESMC or ProtT5, mean-pooled only. Cheap check that the
      conclusion is not an ESM-2 artifact.
- [ ] **3.5 Arm E — groove instead of pseudosequence.** Re-run the best-performing arm
      with `hla_seq` (182-aa alpha1/alpha2 domain) in place of `hla_pseudoseq`. The
      pseudosequence is 34 non-contiguous residues and may be out of distribution for a
      pLM; the groove domain is a real protein domain and is not. This is a documented
      risk to P2 and this arm is the mitigation.

Run embeddings on Modal using the existing `scripts/modal_embed.py` pattern. **Record
wall-clock GPU seconds and peak memory for every arm** — this feeds the compute plot and
the brief names engineering cost as a judging criterion.

- [ ] **3.6** Orchestrator `src/run_benchmark.py` writing
      `results/benchmark/per_fold.csv` (`arm, split_regime, fold, metric, value`) and
      `results/benchmark/per_row.csv` (`row_index, arm, split_regime, y_true, y_pred`).
      Everything downstream reads these two files.

---

## Phase 4 — the headline analysis (~1 h)

- [ ] **4.1** Table: metric × arm × split regime, mean ± sd over folds. Does Arm A lose
      on peptide-split (P1)? Does any arm win on allele-split (P2)?
- [ ] **4.2** **Headline figure.** Held-out-allele Spearman per arm, stratified by
      `max_identity_to_train_allele` band and by `n_train_rows` for that allele. H1
      predicts the pLM advantage is largest where both are smallest.
- [ ] **4.3** Learning curve (P3): subsample training rows at 10/25/50/100% and plot the
      arm-vs-baseline gap against training size.
- [ ] **4.4** Compute plot: Spearman on y, GPU-seconds (or dollars) on x, one point per
      arm. Almost no other team will produce this.
- [ ] **4.5** Write `reports/benchmark_findings.md` stating, in order: what each
      prediction predicted, what happened, and whether H1 survives. **A negative result
      is an acceptable and explicitly welcomed outcome — report it straight.**

---

## Phase 5 — demo (~2 h, do not skip)

Worth 20% of the judging rubric and there is currently nothing on screen.

- [ ] **5.1** `app.py`, Gradio. Input: a peptide (9-mer) and an allele picked from a
      dropdown. Output card showing:
      - predicted half-life with an uncertainty band (fold spread is sufficient),
      - the stability tier badge (`≥6 h` / `2–6 h` / `<2 h`),
      - each arm's prediction side by side,
      - **how many training measurements exist for that allele**, and the identity of its
        nearest better-measured neighbour,
      - **whether the pair appears in any known model's training data** — reuse the
        overlap logic in `scripts/overlap_report.py`.
- [ ] **5.2** Pre-load two example states: one well-measured allele, one under-measured
      allele. The demo moment is switching between them and watching the baseline and
      the foundation-model arms separate.

---

## Phase 6 — stretch, only if Phase 5 is green

- [ ] **6.1** Structure arm: Boltz-2 or Chai-1 on 50–100 complexes; correlate ipTM/PAE
      against measured half-life. The brief explicitly names internal confidence metrics.
- [ ] **6.2** Contamination case study: run a released SPEARMINT checkpoint on its own
      reported benchmark, then on pairs it has not seen, and show both numbers.
- [ ] **6.3** Merge the corrected `audit()` from `agenthandoff.md` Appendix A into
      `check_stability_data.py` — the shipped version still has the original comment
      screen and reproduces 286 unflagged rows instead of 6,101. Add a regression test
      asserting 6,101 / 5,815 / 6.

---

## Phase 7 — ship (~1 h, reserve it)

- [ ] **7.1** `README.md`: the research question, H1 and the four predictions, how to
      reproduce (`pip install -r requirements.txt` → `python -m src.run_benchmark`),
      the result table, and the limitations.
- [ ] **7.2** Confirm every results CSV is committed and the repo clones-and-runs from
      scratch in a fresh directory.
- [ ] **7.3** Limitations section, stated plainly: zeros are likely left-censored and are
      modelled as exact; `(C67S)` constructs excluded; 9-mers and class I only;
      retrospective, no laboratory claim; one head architecture.

---

## 3. Decisions already made — do not relitigate

- **Zeros stay in.** ~20% of rows have `thalf_hours == 0`. `log1p(0) = 0`, which is
  well-defined, and Spearman handles the ties. Excluding them would restrict the task to
  peptides already known to bind — the easier and less useful half of the problem, and
  it would make the tier metric meaningless. Note the censoring in limitations.
- **Primary metric is Spearman ρ**, because the operational use is ranking candidate
  peptides for a given allele.
- **Primary HLA representation is `hla_pseudoseq`**; `hla_seq` is Arm E.
- **Engineered `(C67S)` alleles are excluded** from the main analysis, counted and
  reported.
- **The old IEDB active-learning work stays in the repo** under `scripts/`. It is one
  slide in the pitch: diversity-based acquisition lost to random in 10/10 seeds, and the
  measured reason is the feature-geometry defect in constraint 3 above. Do not delete it
  and do not refactor it.

---

## 4. Definition of done

Minimum viable submission, in priority order:

1. Phases 0–2 complete: the data loads, splits are locked on disk, the baseline runs on
   both split regimes with Spearman reported.
2. Arm A and Arm B evaluated on both regimes. This alone tests P1, P2 and P4.
3. The headline stratified figure (4.2) and `reports/benchmark_findings.md`.
4. The Gradio demo running locally.
5. README and committed results.

Arms C, D, E and all of Phase 6 are upside. **Do not start Phase 6 until Phase 5 runs.**
