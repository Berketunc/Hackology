# checkv1.md — next steps from commit `061a545`

State: the matched-head extension is in and verified. 26 new fits, zero
`training_row_ids_sha256` mismatches against `blosum_nn`, identical macro
eligibility across all seven arms, and every number in
`reports/benchmark_findings.md` reproduces from `per_fold.csv`. The negative
result survives the control and is now supported under two downstream
procedures of very different capacity — the strongest version of this claim.

One substantive item remains. Everything after it is disclosure and hygiene.

---

## 1. Patience sensitivity on the allele folds  (≈10 min, 18 fits)

### Why

Three of five `esm2_joint_nn` allele folds stopped at `best_epoch` ≤ 2, meaning
inner validation loss never improved across 15 consecutive epochs after the
first. Those models are untrained, and the deficit concentrates in them:

| arm | Δ vs `blosum_nn`, folds with best epoch ≤2 | Δ, folds that trained longer |
|---|---:|---:|
| `esm2_joint_nn` | −0.181 (n=3) | −0.079 (n=2) |
| `esm2_mean_nn` | −0.217 (n=1) | −0.139 (n=4) |

Root cause is identifiable: `fit_baseline_nn` splits inner validation with
`GroupShuffleSplit` on the same key as the outer fold, so in the allele regime
it validates on entirely unseen alleles — a noisy target that the
high-dimensional arms misfire on. Patience is 15 against a 150-epoch budget.

Treat the stratification as suggestive, not established (n=3 vs n=2, and it is
post-hoc on a variable correlated with the outcome). But `best_epoch` = 1 is a
diagnostic of the stopping rule, not of the representation, so the allele
number is the weakest in your set and a judge who spots it will say so.

The peptide regime needs none of this: epochs 7–56, deficit consistent at
≈−0.24 across all five folds.

### 1a. Parameterise patience — default preserves everything

`src/models.py` line 29 and line 64:

```python
def fit_baseline_nn(X_train, y_train, seed=0, groups=None, max_epochs=150, patience=15):
```

```python
        if stale >= patience:
            break
```

The default is unchanged, so every existing call site and the whole locked
benchmark stay bit-identical on rerun. Do not change the default.

### 1b. Run it as a separate, non-destructive study

Do **not** add new arms to `ARMS` or write into
`results/benchmark/jobs/` — the job stems have no slot for a procedure variant
and you would overwrite prespecified fits. New script,
`scripts/patience_sensitivity.py`:

```python
"""Post-hoc patience sensitivity for the allele-regime MLP fits.

Triggered by observing best_epoch <= 2 in the prespecified runs; exploratory,
and reported as such. Writes to its own directory; the locked benchmark under
results/benchmark/jobs/ is never touched.
"""
import json, time
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits

from src.config import RESULTS
from src.splits import prepare, split_indices
from src.features import blosum_pairs, load_embedding_arm
from src.models import fit_baseline_nn
from src.evaluate import metrics

OUT = RESULTS / 'benchmark/patience_sensitivity'
PATIENCE = 30
REGIMES = ['allele', 'allele_strict']

def features(arm, df):
    if arm == 'blosum_nn':
        return blosum_pairs(df)
    return load_embedding_arm(arm.replace('_nn', ''), df)

def main():
    df = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, scores = [], []
    for arm in ['blosum_nn', 'esm2_mean_nn', 'esm2_joint_nn']:
        X = features(arm, df)
        for regime in REGIMES:
            for fold, train, test in split_indices(df, regime):
                start = time.perf_counter()
                xt, xv = np.asarray(X[train]), np.asarray(X[test])
                groups = df.iloc[train]['allele'].to_numpy()
                with threadpool_limits(limits=4):
                    model = fit_baseline_nn(xt, df.y.iloc[train], seed=fold,
                                            groups=groups, patience=PATIENCE)
                    pred = model.predict(xv)
                seconds = time.perf_counter() - start
                r = pd.DataFrame(dict(row_index=df.index[test],
                    allele=df.allele.iloc[test].to_numpy(), arm=arm,
                    split_regime=regime, fold=fold, patience=PATIENCE,
                    y_true=df.y.iloc[test].to_numpy(), y_pred=pred))
                rows.append(r)
                m = metrics(r.y_true, r.y_pred, r.allele)
                scores.append(dict(arm=arm, split_regime=regime, fold=fold,
                    patience=PATIENCE, n_train=len(train),
                    best_epoch=model.best_epoch_, seconds=seconds, **m))
                print(f'{arm} {regime} f{fold} ep={model.best_epoch_} '
                      f"macro={m['macro_within_allele_spearman']:.4f}", flush=True)
                del xt, xv, model
    pd.concat(rows, ignore_index=True).to_csv(OUT / 'per_row.csv', index=False)
    pd.DataFrame(scores).to_csv(OUT / 'per_fold.csv', index=False)
    json.dump(dict(patience=PATIENCE, baseline_patience=15, regimes=REGIMES,
                   arms=['blosum_nn','esm2_mean_nn','esm2_joint_nn'],
                   status='post-hoc sensitivity; triggered by best_epoch<=2 in '
                          'prespecified allele fits; not a confirmatory result'),
              open(OUT / 'design.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
```

```bash
python -m scripts.patience_sensitivity
```

Run `esm2_joint_nn` last if memory is tight — 11,520 dims, ~1 GB per training
slice, roughly tripled by the scaler and the torch tensor.

### 1c. Non-negotiable: `blosum_nn` is in the rerun

Changing the stopping rule for only the pLM arms reintroduces exactly the
procedure mismatch the extension was built to remove, in the opposite
direction. All three arms, same patience, same folds, or the comparison is
worthless.

### 1d. Acceptance checks

- 18 rows in `patience_sensitivity/per_fold.csv` (3 arms × [5 allele + 1 strict]).
- No `best_epoch` ≤ 2 remaining. If any persist at patience 30, patience is not
  the cause — say so and stop; don't escalate to 60 hunting for an outcome.
- `n_train` per regime/fold matches the main run.

### 1e. How to report it

Paired deltas vs `blosum_nn` at patience 30, beside the patience-15 values, as
a **robustness check** — not a replacement for the headline. The main table
stays the prespecified procedure. Label the section post-hoc and say what
triggered it. Then one of:

- **Gap narrows materially** → "the allele-regime deficit is sensitive to the
  early-stopping rule; under longer training it is X [CI]. The peptide-regime
  deficit is not sensitive." Honest, and it strengthens the peptide result by
  contrast.
- **Gap holds** → you have closed the last objection to the headline. Say so.

Either outcome is reportable. Pick one knob only — do not also try a
non-grouped inner validation split; two post-hoc variants on one observation
invites a multiplicity objection you cannot answer with six folds.

---

## 2. Add `best_epoch` to the per-fold tables  (≈5 min)

It is already in every job sidecar. Surface it in the report's per-fold or
appendix table with one sentence noting the allele-regime values and what you
did about them. A reader who finds `best_epoch` = 1 with no comment will
assume you did not look.

---

## 3. Two sentences for limitations  (≈5 min)

**Input truncation.** `hla_seq` is the 182-residue α1/α2 domain — no α3 domain,
no β2-microglobulin. That is what the organisers supplied and what the brief
specifies, but it means the pLM arms embed a domain fragment rather than a
native chain, which is a plausible contributing reason frozen ESM-2
underperforms. This bears directly on your conclusion and is currently
unstated; line 129 mentions a "groove-domain" study as unrun, which is not the
same point.

**Why no structure-prediction arm.** You list it as unrun. Add the reason:
AF3-class predictors over 28,166 complexes is days of GPU time, and the
nearest relevant output — Boltz-2's affinity head — predicts binding affinity,
which the brief itself distinguishes from stability (stability is dominated by
off-rate; affinity folds in on-rate too). So it is a proxy with a known
mismatch to the target, not a drop-in comparator. Declining it for stated
reasons reads as judgement; listing it bare reads as an omission.

---

## 4. Data provenance — verified, keep the receipt  (0 min, already done)

I pulled the organisers' source spreadsheet directly. It has exactly one tab
(`gid=1629690744`), 28,166 rows × 5 columns, and after sorting it is
content-identical to `data/processed/rasmussen_all.csv`. All brief-stated
properties hold on the source: all 9-mers, 34-residue pseudosequence, three
`(C67S)` engineered constructs, 75 alleles, one `hla_seq` per allele, no
duplicate (allele, peptide) pairs, 182-residue domain. 20.2% of labels are
exactly 0.0 and there are no HLA-C rows — both already disclosed in your report.

The "data structure is wrong" claim does not survive contact with the source.
If it resurfaces, that paragraph is the answer. If whoever raised it meant
something narrower — the splits, or the primary metric — that is a different
and possibly substantive argument worth hearing on its specifics.

---

## 5. Submission hygiene  (≈10 min)

- Move `fix.md`, `fix2.md`, `agenthandoff.md` and the web app out of the repo
  root into `notes/` or `archive/`.
- README line: the v2 ESM-2 feature arrays (`esm2_mean.npy`, `esm2_joint.npy`,
  `*_row_ids.npy`) are not committed — only the `_meta.json` sidecars — so
  reproduction needs the local cache or re-extraction via
  `scripts/modal_benchmark_embed.py`. Without this someone will conclude the
  benchmark is unreproducible.
- Check the figure regenerates with seven arms in `LABELS`/`COLORS`.

---

## Do not start

A diffusion / structure-prediction arm, an inverse-folding arm, a second pLM
checkpoint, masked-likelihood scoring, P3 at matched MLP across budgets, or any
new GPU extraction. Each is a week of work and none is an evening's. They are
named in limitations; that is the correct treatment with hours left.

---

## Checklist

- [ ] Commit before touching anything
- [ ] `src/models.py`: add `patience=15` parameter, use it at the `stale` check, default unchanged
- [ ] `scripts/patience_sensitivity.py` added and run (18 fits, 3 arms incl. `blosum_nn`)
- [ ] No `best_epoch` ≤ 2 remains at patience 30 — or stated plainly if some do
- [ ] Robustness section written, labelled post-hoc, main table unchanged
- [ ] `best_epoch` surfaced in a per-fold table with one explanatory sentence
- [ ] α1/α2 truncation sentence added to limitations
- [ ] Structure-prediction omission given its reason (affinity ≠ stability, compute)
- [ ] Repo root tidied; README note on uncommitted feature arrays
- [ ] Figure regenerates with all seven arms
