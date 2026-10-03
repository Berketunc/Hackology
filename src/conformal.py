"""Allele-conditional split conformal intervals for previously unseen peptides.

Peptides are disjoint across fitting, calibration and evaluation. Within an allele,
each calibration observation is a distinct peptide; no pooled row-independence or
simultaneous coverage across alleles is assumed. No calibration labels tune the NN.
"""
import hashlib
import json
from decimal import Decimal, ROUND_CEILING
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from .config import DATA, RESULTS
from .features import blosum_pairs
from .models import fit_baseline_nn
from .splits import prepare, split_indices, write_locked, _write_assignments

OUT = RESULTS / 'conformal'
MODEL_DIR = RESULTS / 'models/conformal'
ALPHA = .05
CALIBRATION_FRACTION = .25


def conformal_cutoff(scores, alpha=ALPHA):
    """Finite-sample order statistic; +infinity if calibration is insufficient."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Calibration scores must be finite nonnegative scalars.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one.')
    # Decimal avoids ceil(19.000000000000004) at exact rank boundaries.
    rank = int((Decimal(len(scores) + 1) * (1 - Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return (float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else float('inf')), rank


def hour_interval(prediction, cutoff):
    """Use the same nonnegative log predictor for scores, centers and intervals."""
    center = np.maximum(0., np.asarray(prediction, dtype=float))
    cutoff = np.asarray(cutoff, dtype=float)
    if np.isnan(cutoff).any() or (cutoff < 0).any():
        raise ValueError('Invalid conformal cutoff.')
    return np.expm1(center), np.expm1(np.maximum(0., center - cutoff)), np.expm1(center + cutoff)


def calibration_table(rows, train_counts, alleles, alpha=ALPHA):
    assert not rows.duplicated(['allele', 'peptide']).any()
    records = []
    for allele in sorted(alleles):
        g = rows[rows.allele == allele]
        scores = np.abs(g.y_true.to_numpy() - np.maximum(0., g.y_pred.to_numpy()))
        q, rank = conformal_cutoff(scores, alpha)
        n_train = int(train_counts.get(allele, 0))
        status = 'available' if np.isfinite(q) else 'insufficient_calibration'
        if not n_train:
            status, q = 'unseen_allele', float('inf')
        records.append(dict(allele=allele, n_train_rows=n_train, n_calibration=len(g),
                            quantile_rank=rank, q_log=q, status=status, nominal_coverage=1-alpha))
    return pd.DataFrame(records)


def lock_partitions(df):
    """Write every fit/calibration/test assignment before fitting any model."""
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for fold, outer_train, test in split_indices(df, 'peptide'):
        fit_rel, cal_rel = next(GroupShuffleSplit(n_splits=1, test_size=CALIBRATION_FRACTION,
            random_state=1000+fold).split(outer_train, groups=df.iloc[outer_train].peptide))
        fit, cal = outer_train[fit_rel], outer_train[cal_rel]
        for a, b in [(fit, cal), (fit, test), (cal, test)]:
            assert not set(df.iloc[a].peptide) & set(df.iloc[b].peptide)
        role = np.full(len(df), 'test', dtype=object)
        role[fit], role[cal] = 'fit', 'calibration'
        path = OUT / f'partition_fold{fold}.csv'
        _write_assignments(path, pd.DataFrame(dict(row_index=df.index, role=role)))
        parts.append((fold, fit, cal, test))
    write_locked(OUT/'design.json', dict(version=1, arm='blosum_nn', alpha=ALPHA,
        calibration_fraction_of_outer_training=CALIBRATION_FRACTION,
        calibration_seed='1000 + outer fold', model_seed='2000 + outer fold',
        outer_partition='existing five peptide-group folds, seed 0',
        calibration='Mondrian split conformal: separate absolute log residual quantile per allele',
        predictor='max(0, BLOSUM MLP log prediction); same 256/64 NN and grouped inner early stopping',
        finite_sample_rank='ceil((n_calibration + 1) * (1-alpha)); +infinity if rank > n',
        missing_support='No finite interval for insufficient calibration or unseen training allele; no pooled fallback',
        target='Recorded half-life of a new peptide on an allele present in the fit set',
        assumption='Exchangeability of distinct peptide observations within each allele; not simultaneous allele/panel coverage',
        deployment='For any previously measured peptide choose its outer test fold, including new allele pairings; otherwise fixed fold 0. Never average calibrated fits.',
        evaluation='Retrospective held-out evaluation on previously studied dataset; no interval selection using test labels',
        dataset_sha256=json.loads((DATA/'manifests/rasmussen_manifest.json').read_text())['sha256'],
        partition_hashes={str(f):hashlib.sha256((OUT/f'partition_fold{f}.csv').read_bytes()).hexdigest() for f,*_ in parts}))
    return parts


def nearest_peptide_identity(peptides, training_peptides):
    """Exact positional identity to any fitting peptide; sequences are all 9-mers."""
    train = np.array([list(s) for s in sorted(set(training_peptides))], dtype='S1')
    unique = sorted(set(peptides))
    out = {}
    for start in range(0, len(unique), 128):
        seqs = unique[start:start+128]
        batch = np.array([list(s) for s in seqs], dtype='S1')
        identity = (batch[:, None, :] == train[None, :, :]).mean(axis=2).max(axis=1)
        out.update(zip(seqs, identity))
    return pd.Series(peptides).map(out).to_numpy()


def summarize(rows):
    finite = rows[rows.status == 'available']
    return dict(n_test=len(rows), n_finite=len(finite), n_without_finite=len(rows)-len(finite),
        finite_interval_fraction=len(finite)/len(rows),
        coverage_finite=float(finite.covered.mean()),
        mean_width_hours_finite=float(finite.width_hours.mean()),
        median_width_hours_finite=float(finite.width_hours.median()),
        coverage_including_unbounded=float(rows.covered.mean()), nominal_coverage=1-ALPHA)


def run():
    df = prepare()
    parts = lock_partitions(df)
    X = blosum_pairs(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    design_hash = hashlib.sha256((OUT/'design.json').read_bytes()).hexdigest()
    outputs, calibration_outputs, score_outputs, fit_audits = [], [], [], []
    for fold, fit, cal, test in parts:
        path = MODEL_DIR/f'fold{fold}.joblib'
        if path.exists():
            bundle = joblib.load(path)
            assert bundle['design_hash'] == design_hash, 'Stale conformal model; use a fresh output directory.'
            for key, pos in [('fit_row_ids', fit), ('calibration_row_ids', cal), ('test_row_ids', test)]:
                np.testing.assert_array_equal(bundle[key], df.index[pos])
            model = bundle['model']
        else:
            print(f'FIT conformal fold={fold} fit={len(fit)} calibration={len(cal)} test={len(test)}', flush=True)
            with threadpool_limits(limits=4):
                model = fit_baseline_nn(X[fit], df.y.iloc[fit], seed=2000+fold,
                                       groups=df.peptide.iloc[fit].to_numpy())
        with threadpool_limits(limits=4):
            cal_pred, test_pred = model.predict(X[cal]), model.predict(X[test])
        c = df.iloc[cal][['allele','peptide','y']].rename(columns={'y':'y_true'}).reset_index()
        c['y_pred'], c['fold'] = cal_pred, fold
        c['absolute_error_log'] = np.abs(c.y_true - np.maximum(0., c.y_pred))
        scores = calibration_table(c, df.iloc[fit].groupby('allele').size(), df.allele.unique())
        scores['fold'] = fold
        r = df.iloc[test][['allele','peptide','thalf_hours','y']].rename(columns={'y':'y_true'}).reset_index()
        r['y_pred'], r['fold'] = test_pred, fold
        r = r.merge(scores, on=['allele','fold'], validate='many_to_one')
        r['predicted_hours'], r['lower_hours'], r['upper_hours'] = hour_interval(r.y_pred, r.q_log)
        r['covered'] = (r.thalf_hours >= r.lower_hours) & (r.thalf_hours <= r.upper_hours)
        r['width_hours'] = r.upper_hours - r.lower_hours
        r['nearest_fit_peptide_identity'] = nearest_peptide_identity(r.peptide, df.iloc[fit].peptide)
        bundle = dict(model=model, design_hash=design_hash, calibration=scores,
                      fit_row_ids=df.index[fit].to_numpy(), calibration_row_ids=df.index[cal].to_numpy(),
                      test_row_ids=df.index[test].to_numpy(), fold=fold)
        joblib.dump(bundle, path)
        fit_audits.append(dict(fold=fold, seed=2000+fold, calibration_seed=1000+fold,
            fit_rows=len(fit), calibration_rows=len(cal), test_rows=len(test),
            inner_training_rows=model.inner_train_size_, inner_validation_rows=model.inner_validation_size_,
            best_epoch=model.best_epoch_, design_sha256=design_hash,
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        outputs.append(r); calibration_outputs.append(scores); score_outputs.append(c)
        print(dict(fold=fold, **summarize(r)), flush=True)
    rows = pd.concat(outputs, ignore_index=True)
    assert len(rows) == len(df) and not rows.row_index.duplicated().any()
    rows.to_csv(OUT/'per_row.csv', index=False)
    pd.concat(calibration_outputs, ignore_index=True).to_csv(OUT/'calibration.csv', index=False)
    pd.concat(score_outputs, ignore_index=True).to_csv(OUT/'calibration_scores.csv', index=False)
    pd.DataFrame(fit_audits).to_csv(OUT/'fit_audit.csv', index=False)
    summary = [dict(scope='overall', fold=-1, **summarize(rows))]
    summary += [dict(scope='fold', fold=f, **summarize(g)) for f,g in rows.groupby('fold')]
    pd.DataFrame(summary).to_csv(OUT/'coverage.csv', index=False)
    per_allele = [dict(allele=a, **summarize(g)) for a,g in rows.groupby('allele')]
    pd.DataFrame(per_allele).to_csv(OUT/'coverage_by_allele.csv', index=False)
    rows['identity_band'] = pd.cut(rows.nearest_fit_peptide_identity, [0,.6,.8,1.00001], right=False,
                                  labels=['<60%', '60–80%', '80–100%'])
    pd.DataFrame([dict(identity_band=str(b), **summarize(g)) for b,g in rows.groupby('identity_band', observed=True)]).to_csv(OUT/'coverage_by_distance.csv', index=False)
    rows[rows.status != 'available'].to_csv(OUT/'unavailable_intervals.csv', index=False)


@lru_cache(maxsize=5)
def load_bundle(fold):
    bundle = joblib.load(MODEL_DIR/f'fold{fold}.joblib')
    assert bundle['design_hash'] == hashlib.sha256((OUT/'design.json').read_bytes()).hexdigest()
    return bundle


def predict_interval(peptide, allele, df):
    """Route by peptide, not exact pair, so a new pairing cannot leak its peptide."""
    same_peptide = df[df.peptide == peptide]
    fold = 0
    if len(same_peptide):
        folds = pd.read_csv(RESULTS/'splits/peptide_folds.csv').set_index('row_index').fold
        assigned = folds.loc[same_peptide.index].unique()
        assert len(assigned) == 1
        fold = int(assigned[0])
    if not (MODEL_DIR/f'fold{fold}.joblib').exists():
        return None
    bundle = load_bundle(fold)
    for key in ['fit_row_ids','calibration_row_ids']:
        assert peptide not in set(df.loc[bundle[key]].peptide)
    calibration = bundle['calibration'].set_index('allele').loc[allele]
    pseudo = df.loc[df.allele == allele, 'hla_pseudoseq'].iloc[0]
    x = blosum_pairs(pd.DataFrame(dict(peptide=[peptide], hla_pseudoseq=[pseudo])))
    with threadpool_limits(limits=4):
        prediction = float(bundle['model'].predict(x)[0])
    center, low, high = hour_interval(prediction, calibration.q_log)
    coverage = pd.read_csv(OUT/'coverage.csv')
    validation = coverage[(coverage.scope == 'fold') & (coverage.fold == fold)].iloc[0]
    allele_coverage = pd.read_csv(OUT/'coverage_by_allele.csv').set_index('allele').loc[allele]
    return dict(center=float(center), low=float(low), high=float(high), fold=fold,
                n_calibration=int(calibration.n_calibration), n_train=int(calibration.n_train_rows),
                status=calibration.status, test_coverage=float(validation.coverage_finite),
                n_test_finite=int(validation.n_finite),
                allele_coverage=float(allele_coverage.coverage_finite),
                allele_n_test=int(allele_coverage.n_finite))


if __name__ == '__main__':
    run()
