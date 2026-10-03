"""Run locked folds; resume individual completed arm/fold jobs."""
import argparse
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .config import ROOT, RESULTS, DATA, CHECKPOINT
from .splits import prepare, split_indices
from .features import blosum_pairs, blosum_encode, onehot_allele, load_embedding_arm, embedding_arm_available
from .models import fit_ridge, fit_baseline_nn
from .evaluate import metrics

OUT = RESULTS / 'benchmark'
ARMS = ['blosum_nn', 'blosum_ridge', 'onehot_ridge', 'esm2_mean', 'esm2_joint']


def lock_design():
    design = dict(seed=0, folds=5, version=2, split_regimes=['peptide', 'allele', 'locus', 'allele_strict'], arms=ARMS,
        target='log1p(hours)', primary_metric='macro_within_allele_spearman', companion_metric='pooled_spearman',
        minimum_test_rows=20, minimum_distinct_nonzero_half_lives=10, high_zero_fraction=.5,
        reference_baseline='blosum_nn', illustrative_floor='onehot_ridge',
        paired_ci='95% paired t interval over folds; per-locus and strict fold use paired allele bootstrap (seed 0; 10000 resamples)',
        metric_eligibility_scope='Within each evaluation fold, based on test labels; exclusions exported for every arm/budget/fold',
        original_fits='200 v1 jobs reused unchanged from results/archive/benchmark_v1; v2 macro/CI reanalysis is retrospective',
        seed_schedule='outer split seed=0; model/PCA/inner-validation/nested-subset seed=outer fold',
        strict_budget='full retained strict training set only; compared with permissive fold0, not an isolated causal estimate of bleed',
        locus_budget='full training rows for each observed locus holdout; no C locus is present',
        ridge_head='StandardScaler -> randomized PCA (up to 256) -> RidgeCV LOO; all fit on outer training only',
        alphas=np.logspace(-2, 5, 8).tolist(), nn='BLOSUM62; 256/64 ReLU; AdamW lr=.001 wd=.01; max150 epochs; patience15; training-only group validation 15%',
        checkpoint=CHECKPOINT, hidden_size=1280, linker='GGGG',
        representation='A: separate mean peptide/pseudoseq; B: 9 peptide hidden states of peptide+GGGG+pseudoseq',
        learning_fractions=[.1, .25, .5, 1.0], identity_bands=[0, .8, .9, .95, 1.0001],
        pca_note='Same 256-component preprocessing rule and Ridge head for all Ridge arms; NN is a separate challenge baseline.',
        inference_note='Joint sequence encoding permits cross-token attention; it does not establish a physical complex representation.',
        data_sha256=json.loads((DATA / 'manifests/rasmussen_manifest.json').read_text())['sha256'],
        split_hashes={r: hashlib.sha256((RESULTS / f'splits/{r}_folds.csv').read_bytes()).hexdigest() for r in ['peptide','allele','locus']})
    design['strict_split_hash'] = hashlib.sha256((RESULTS / 'splits/allele_folds_strict_fold0.csv').read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'design.json'
    if path.exists():
        assert json.loads(path.read_text()) == design, 'Locked benchmark design differs'
    else:
        path.write_text(json.dumps(design, indent=2) + '\n')


def run(arms, fractions=(1.0,), regimes=('peptide','allele','locus','allele_strict')):
    df = prepare()
    lock_design()
    missing = [arm for arm in arms if arm.startswith('esm2_') and not embedding_arm_available(arm)]
    if missing:
        from scripts.modal_benchmark_embed import main as extract_embeddings
        extract_embeddings()
    cache = {}
    for arm in arms:
        if arm.startswith('esm2_'):
            cache[arm] = load_embedding_arm(arm, df)
        elif arm.startswith('blosum'):
            cache[arm] = blosum_pairs(df)
    for regime in regimes:
        for fold, full_train, test in split_indices(df, regime):
            order = np.random.default_rng(fold).permutation(full_train)
            for fraction in (fractions if regime in ['peptide','allele'] else [1.0]):
                train = full_train if fraction == 1 else order[:max(10, int(len(order)*fraction))]
                for arm in arms:
                    stem = f'{arm}_{regime}_{fold}_{fraction:g}'
                    path = OUT / 'jobs' / f'{stem}.csv'
                    saved_model = RESULTS / 'models' / f'{arm}_{regime}_{fold}.joblib'
                    if path.exists() and (fraction != 1 or saved_model.exists()):
                        continue
                    print(f'RUN {stem} train={len(train)} test={len(test)}', flush=True)
                    start = time.perf_counter()
                    encoder = None
                    if arm == 'onehot_ridge':
                        a, encoder = onehot_allele(df.iloc[train])
                        b, _ = onehot_allele(df.iloc[test], encoder)
                        xt = np.column_stack([np.stack(df.iloc[train].peptide.map(blosum_encode)), a])
                        xv = np.column_stack([np.stack(df.iloc[test].peptide.map(blosum_encode)), b])
                    else:
                        xt, xv = np.asarray(cache[arm][train]), np.asarray(cache[arm][test])
                    with threadpool_limits(limits=4):
                        if arm == 'blosum_nn':
                            model = fit_baseline_nn(xt, df.y.iloc[train], seed=fold, groups=df.iloc[train]['peptide' if regime=='peptide' else 'allele'].to_numpy())
                        else:
                            model = fit_ridge(xt, df.y.iloc[train], seed=fold)
                        pred = model.predict(xv)
                    seconds = time.perf_counter() - start
                    rows = pd.DataFrame(dict(row_index=df.index[test], allele=df.allele.iloc[test].to_numpy(),
                        arm=arm, split_regime=regime, fold=fold, fraction=fraction,
                        n_train=len(train), y_true=df.y.iloc[test].to_numpy(), y_pred=pred))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    rows.to_csv(path, index=False)
                    stats = dict(arm=arm, split_regime=regime, fold=fold, fraction=fraction,
                                 n_train=len(train), seed=fold,
                                 training_row_ids_sha256=hashlib.sha256(df.index.to_numpy()[train].astype('<i8').tobytes()).hexdigest(),
                                 fit_predict_cpu_wall_seconds=seconds,
                                 alpha=float(model.named_steps['ridge'].alpha_) if arm != 'blosum_nn' else None,
                                 best_epoch=getattr(model, 'best_epoch_', None))
                    (path.with_suffix('.json')).write_text(json.dumps(stats, indent=2))
                    if fraction == 1:
                        target = RESULTS / 'models' / f'{arm}_{regime}_{fold}.joblib'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(dict(model=model, encoder=encoder, row_indices=df.index[train].to_numpy()), target)
                    print(metrics(rows.y_true, rows.y_pred, rows.allele), flush=True)
                    # Summarise once per invocation; full audit is expensive across all budgets.
    aggregate()


def aggregate():
    files = sorted((OUT / 'jobs').glob('*.csv'))
    if not files:
        return
    rows = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df = prepare()
    truth=df[['allele','thalf_hours','locus']].reset_index()
    rows=rows.merge(truth,on=['row_index','allele'],validate='many_to_one')
    assert np.allclose(rows.y_true,np.log1p(rows.thalf_hours),atol=1e-14)
    rows=rows.rename(columns={'thalf_hours':'y_hours'})
    full = rows[rows.fraction == 1]
    full.to_csv(OUT / 'per_row.csv.tmp', index=False)
    (OUT / 'per_row.csv.tmp').replace(OUT / 'per_row.csv')
    scores = []
    audits=[]
    from .evaluate import allele_metrics
    for key, group in rows.groupby(['arm','split_regime','fold','fraction']):
        universe=df.allele.unique() if key[1]=='peptide' else None
        audit=allele_metrics(group,universe)
        for col,val in zip(['arm','split_regime','fold','fraction'],key): audit[col]=val
        audits.append(audit)
        for metric, value in metrics(group.y_true, group.y_pred, group.allele,universe).items():
            scores.append(dict(zip(['arm','split_regime','fold','fraction'], key), metric=metric, value=value))
    audit=pd.concat(audits,ignore_index=True)
    audit.to_csv(OUT/'allele_metric_audit.csv',index=False)
    audit[~audit.include_in_macro].to_csv(OUT/'excluded_alleles.csv',index=False)
    audit[audit.high_zero_fraction].to_csv(OUT/'high_zero_allele_auc.csv',index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / 'per_fold.csv', index=False)
    scores.to_csv(OUT / 'learning_curve.csv', index=False)
    pd.DataFrame([json.loads(p.read_text()) for p in sorted((OUT/'jobs').glob('*.json'))]).to_csv(OUT/'fit_compute.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--fractions', nargs='+', type=float, default=[1.0])
    parser.add_argument('--regimes', nargs='+', choices=['peptide','allele','locus','allele_strict'], default=['peptide','allele','locus','allele_strict'])
    args = parser.parse_args()
    assert all(f in [.1,.25,.5,1.] for f in args.fractions)
    run(args.arms, args.fractions, args.regimes)
