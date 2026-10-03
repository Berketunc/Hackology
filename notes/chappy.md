# check.md — what to do next

Runbook for the remaining hours. Ordered by value, each step time-boxed.
Steps 1 and 2 are independent; start 1 first because it takes wall time.

**Correction to my earlier review:** I claimed the noise ceiling was
estimable from `rasmussen_overlap.csv` / `conflicting_measurements.csv`. It
is not. The first holds SPEARMINT split-membership flags; the second belongs
to the IEDB pilot, a different dataset. `src/data.py` asserts
`not df.duplicated(['allele','peptide']).any()` — the Rasmussen file has no
repeat measurements. See step 2 for what to do instead.

---

## 1. Run the ESM-2 MLP arms  (≈20 min wall, the whole point)

This is the missing cell of the 2×2. Without it, "ESM-2 loses to the
sequence-aware baseline" conflates representation with head capacity: on your
own saved folds, MLP-minus-Ridge on *identical* BLOSUM features is **+0.354**
(peptide) and **+0.164** (allele), while ESM-2 joint minus BLOSUM at a matched
Ridge head is only **−0.043** and **−0.038**.

### 1a. Patch `src/run_benchmark.py` — four edits

**Line 17**, add two arms:

```python
ARMS = ['blosum_nn', 'blosum_ridge', 'onehot_ridge', 'esm2_mean', 'esm2_joint',
        'esm2_mean_nn', 'esm2_joint_nn']
```

**In `run()`**, the availability check and the cache both need to strip the
suffix so the new arms reuse the existing feature binaries:

```python
# was: missing = [arm for arm in arms if arm.startswith('esm2_') and not embedding_arm_available(arm)]
missing = [arm for arm in arms
           if arm.startswith('esm2_') and not embedding_arm_available(arm.replace('_nn', ''))]
```

```python
    for arm in arms:
        if arm.startswith('esm2_'):
            cache[arm] = load_embedding_arm(arm.replace('_nn', ''), df)   # changed
        elif arm.startswith('blosum'):
            cache[arm] = blosum_pairs(df)
```

**Head selection** — the only functional change:

```python
                    with threadpool_limits(limits=4):
                        if arm.endswith('_nn'):                            # was: arm == 'blosum_nn'
                            model = fit_baseline_nn(xt, df.y.iloc[train], seed=fold,
                                groups=df.iloc[train]['peptide' if regime=='peptide' else 'allele'].to_numpy())
                        else:
                            model = fit_ridge(xt, df.y.iloc[train], seed=fold)
                        pred = model.predict(xv)
```

**Stats row**, so it doesn't reach for a Ridge attribute that isn't there:

```python
                                 alpha=float(model.named_steps['ridge'].alpha_) if not arm.endswith('_nn') else None,
```

### 1b. Patch `src/report.py` — or it will crash

`LABELS` / `COLORS` / `PLMS` are hardcoded dicts and several loops do
`COLORS[arm]` after a `groupby('arm')`. New arms raise `KeyError`. Lines 14–17:

```python
LABELS={'blosum_nn':'BLOSUM MLP (reference)','blosum_ridge':'BLOSUM Ridge',
        'onehot_ridge':'Allele-ID Ridge (floor)','esm2_mean':'ESM-2 mean',
        'esm2_joint':'ESM-2 joint',
        'esm2_mean_nn':'ESM-2 mean + MLP','esm2_joint_nn':'ESM-2 joint + MLP'}
COLORS=dict(zip(LABELS,['#2563eb','#6b7280','#c0a1c9','#ea580c','#059669',
                        '#b45309','#0f766e']))
PLMS=['esm2_mean','esm2_joint','esm2_mean_nn','esm2_joint_nn']
```

### 1c. Run it, one arm at a time

```bash
python -m src.run_benchmark --arms esm2_mean_nn  --fractions 1.0
python -m src.run_benchmark --arms esm2_joint_nn --fractions 1.0
```

Full budget only: 13 folds per arm (5 peptide + 5 allele + 2 locus + 1 strict),
26 fits total. For reference, `blosum_nn` at 860 dims takes 1.6 s per
full-budget fit, with best epoch 3–12 — early stopping fires fast.

**Watch memory on the joint arm.** It is 9×1280 = 11,520 dims, so
`np.asarray(cache[arm][train])` is ~1.05 GB for 22,879 training rows, and
`StandardScaler` plus the torch tensor roughly triple that. On 8 GiB, run the
two arms in separate processes as above — do not pass both to one invocation.
If it still dies, say so rather than switching the joint arm to PCA-256: that
would break head parity with `blosum_nn` and reintroduce the confound you are
trying to remove.

`aggregate()` globs `results/benchmark/jobs/*.csv`, so `summary.csv`,
`per_fold.csv` and `per_row.csv` pick up the new arms automatically. Commit
before you run so the rewrite is revertible.

### 1d. Acceptance checks

- 26 new files in `results/benchmark/jobs/esm2_*_nn_*.csv`, each with a sibling
  `.json` whose `alpha` is `null` and `best_epoch` is an integer.
- `training_row_ids_sha256` for `esm2_mean_nn_<regime>_<fold>_1` must equal the
  value already recorded for `blosum_nn_<regime>_<fold>_1`. If it doesn't, the
  folds didn't align and nothing below is valid.
- `macro_eligible_alleles` per fold must match the existing arms' values.

### 1e. The number that matters

Paired over folds, at full budget, primary metric:

```python
import pandas as pd
from src.evaluate import paired_delta_ci
pf = pd.read_csv('results/benchmark/per_fold.csv')
d = pf[(pf.metric=='macro_within_allele_spearman') & (pf.fraction==1)]
for regime in ['peptide','allele','locus','allele_strict']:
    g = d[d.split_regime==regime]
    for arm in ['esm2_mean_nn','esm2_joint_nn']:
        print(regime, arm, paired_delta_ci(g[g.arm==arm], g[g.arm=='blosum_nn'], by='fold'))
```

Two folds for locus and one for strict, so those intervals stay descriptive —
label them as you already do elsewhere.

---

## 2. The noise ceiling — reframe, don't estimate  (≈15 min)

You cannot estimate it from the supplied data: no duplicate (allele, peptide)
rows exist, and `src/data.py` asserts their absence. So replace the bare
"unknown" with a precise statement of *why* it is unknown, plus whatever
external anchor you can cite:

> The brief states each label averages at least two experiments, but no
> per-replicate values are supplied and the file contains no duplicate
> allele–peptide rows (asserted in `src/data.py`), so assay reproducibility
> cannot be estimated from these data at all. Published reproducibility for
> this scintillation-proximity dissociation assay family [cite Harndahl et al.
> 2012 / Rasmussen et al. 2016 if either reports it] is therefore the only
> available anchor; absent that, every effect size below should be read against
> an unestimated ceiling.

Check the Rasmussen 2016 methods and the Harndahl assay paper for a replicate
correlation or CV figure. If one exists, quote it and you have a scale for
every number in the report. If neither reports it, the paragraph above is
still strictly better than "unknown" — it tells a reader the gap is a property
of the dataset, not an omission on your part.

Do **not** use the IEDB pilot's `conflicting_measurements.csv` as a proxy.
Different dataset, different protocols, cross-study rather than within-assay;
it would measure something else and invite a fair objection.

---

## 3. Rewrite the headline as the decomposition  (≈30 min)

Lead with head-capacity-versus-representation, not pLM-versus-baseline. Which
claim you can make depends on what step 1 returns:

**If ESM-2 + MLP ≈ BLOSUM + MLP** (within ~0.03): the confound is closed and
the negative is now clean. Claim: *frozen ESM-2 features carry no advantage
over BLOSUM encoding for this task, and the large apparent deficit in the
Ridge-headed comparison was head capacity, not representation.* Stronger and
more specific than the current v2 claim.

**If ESM-2 + MLP clearly beats BLOSUM + MLP**: the conclusion inverts. Say so
plainly, lead with it, and note that the v2 report's negative was an artifact
of head choice — caught by your own control. Budget time for this; it is not
the unlikely branch.

**If it lands between**: report the decomposition as the finding. Head capacity
contributes X, representation contributes Y, and the honest answer is that most
of what looked like a pretraining deficit was never about pretraining.

Keep intact: the P1–P4 scaffold, macro-over-alleles with its eligibility rule
and exclusion audit, the strict fold with its "not an isolated causal estimate"
caveat, the disclosure that the v2 metric change followed v1 inspection, tier
AUC for zero-heavy alleles, and the refusal to use NetMHCstabpan as a
comparator. That half of the submission is already strong — don't touch it.

### Add this regardless of outcome

The structural point is worth its own short subsection, because it explains
your P4 result mechanistically rather than descriptively:

> `blosum_ridge` and `onehot_ridge` share an identical BLOSUM peptide block and
> differ only in the HLA block (34-residue pseudosequence vs. one-hot allele
> ID). Their paired difference in macro within-allele Spearman is
> −0.000 [−0.001, +0.001] on peptide holdout and −0.000 [−0.001, +0.000] on
> allele holdout. This is an identity, not a coincidence: the HLA features are
> constant within an allele, a linear head therefore adds them as a per-allele
> offset, and within-allele Spearman is invariant to a per-allele offset. Under
> the primary endpoint every linear arm is effectively a peptide-only model,
> which applies directly to `esm2_mean`, whose MHC half is allele-constant by
> construction. `esm2_joint` is the exception — attention mixes the allele
> context into the peptide positions before pooling, so its features are not
> allele-constant. It is the only Ridge arm that can express a peptide×allele
> interaction, it beats `esm2_mean` in every regime, and it is the only arm
> that goes positive under locus transfer. The extraction that wins is the one
> that survives the metric.

If you run out of time for step 1, **this paragraph plus the matched-head table
is the minimum viable disclosure.** It converts your largest vulnerability into
the most impressive passage in the document, and it takes fifteen minutes.

---

## 4. Demote the locus result  (≈5 min)

Two folds, absolute macro ρ 0.06–0.11, two-fold CI [−0.226, +0.325]. You
already call it highly imprecise in prose and then give it the headline figure.
Pick one. Keep a sentence and the per-locus table in the appendix; take it out
of the headline. Any space it occupies reads as overclaiming to a reader who
has seen the interval.

---

## 5. Tidy the repo root  (≈10 min)

Move `fix.md`, `fix2.md`, `agenthandoff.md`, the pilot/active-learning study
and `app.py` / `src/webapp.py` into `archive/` or `notes/`. Leave `README.md`,
`src/`, `scripts/`, `data/`, `results/benchmark/`, `reports/`. A judge's first
impression is `ls`, and right now it reads as several projects rather than one
controlled study.

One README line worth adding: the v2 ESM-2 feature arrays
(`esm2_mean.npy`, `esm2_joint.npy`, `*_row_ids.npy`) are not committed — only
the `_meta.json` sidecars are. State that reproduction requires either the
local cache or re-extraction via `scripts/modal_benchmark_embed.py`, so nobody
concludes the benchmark is unreproducible.

---

## Do not start

Structure-prediction arms (Boltz-2, Chai-1, Protenix), inverse folding
(ProteinMPNN, ESM-IF), a second pLM checkpoint, masked-likelihood scoring, or
anything needing new GPU extraction. Each is credible as a week of work and
none is credible overnight; a subsampled version would be weaker evidence than
what you already have. Name them in limitations — you largely do — and stop.

---

## Checklist

- [ ] Commit current state before patching
- [ ] `src/run_benchmark.py`: four edits (ARMS, missing-check, cache, head, alpha)
- [ ] `src/report.py`: LABELS / COLORS / PLMS extended
- [ ] `esm2_mean_nn` run, 13 jobs written
- [ ] `esm2_joint_nn` run, 13 jobs written
- [ ] `training_row_ids_sha256` matches `blosum_nn` per regime/fold
- [ ] Paired deltas vs `blosum_nn` computed for all four regimes
- [ ] Noise-ceiling paragraph rewritten (step 2)
- [ ] Headline rewritten as decomposition (step 3)
- [ ] Linear-head invariance subsection added (step 3)
- [ ] Locus result demoted (step 4)
- [ ] Repo root tidied, README note on uncommitted features (step 5)
