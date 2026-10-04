"""Post-hoc patience-30 sensitivity; never replaces locked benchmark jobs.

Triggered by best_epoch <= 2 in the original allele fits. All three MLP
approaches use the same variant; no other setting or evaluation split changes.
"""
import hashlib
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.config import ROOT, RESULTS, REPORTS
from src.splits import prepare, split_indices
from src.features import blosum_pairs, load_embedding_arm
from src.models import fit_baseline_nn
from src.evaluate import PRIMARY, metrics, allele_metrics, paired_delta_ci

BASE = RESULTS / 'benchmark'
OUT = BASE / 'patience_sensitivity'
ARMS = ['blosum_nn', 'esm2_mean_nn', 'esm2_joint_nn']
REGIMES = ['allele', 'allele_strict']
PATIENCE = 30


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def protected_hashes():
    files = [p for p in BASE.rglob('*') if p.is_file() and OUT not in p.parents]
    files += [p for p in (RESULTS/'splits').rglob('*') if p.is_file()]
    files += [REPORTS/'benchmark_table.md', REPORTS/'matched_head_table.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def lock_design():
    path = OUT/'design.json'
    design = dict(patience=30, baseline_patience=15, max_epochs=150, arms=ARMS,
        regimes=REGIMES, fractions=[1.0], expected_fits=18,
        status='Post-hoc sensitivity triggered by best_epoch <= 2 in original allele fits; not confirmatory',
        procedure='Same features, 256/64 MLP, AdamW, allele-grouped inner validation and fold seed; only patience changes',
        stop_rule='Evaluate patience 30 once; persistent early selected epochs are reported, not escalated to 60',
        gpu_extraction=False, base_checkpoint='2df2f32',
        primary_metric=PRIMARY, ci='Five-fold paired t for allele; paired-allele bootstrap conditional on strict fold 0',
        protected_sha256=protected_hashes())
    OUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == design, 'Sensitivity design or protected benchmark changed'
    else:
        path.write_text(json.dumps(design, indent=2)+'\n')
    return design


def features(arm, df):
    return blosum_pairs(df) if arm == 'blosum_nn' else load_embedding_arm(arm.removesuffix('_nn'), df)


def compare(a, b, regime):
    if regime == 'allele':
        return paired_delta_ci(a, b, value=PRIMARY)
    return paired_delta_ci(allele_metrics(a).query('label_eligible'),
                           allele_metrics(b).query('label_eligible'),
                           by='allele', value='spearman', method='bootstrap')


def report(rows, scores):
    old_scores = pd.read_csv(BASE/'per_fold.csv')
    old_scores = old_scores[(old_scores.fraction==1)&(old_scores.metric==PRIMARY)].rename(columns={'value': PRIMARY})
    old_rows = pd.read_csv(BASE/'per_row.csv')
    contrasts=[]
    for patience, sf, rf in [(15,old_scores,old_rows),(30,scores,rows)]:
        for regime in REGIMES:
            frame=sf if regime=='allele' else rf
            g=frame[frame.split_regime==regime]
            for arm in ARMS[1:]:
                contrasts.append(dict(patience=patience,arm=arm,baseline='blosum_nn',split_regime=regime,
                                      **compare(g[g.arm==arm],g[g.arm=='blosum_nn'],regime)))
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(OUT/'paired_comparisons.csv',index=False)
    shifts=[]
    for regime in REGIMES:
        a=scores if regime=='allele' else rows
        b=old_scores if regime=='allele' else old_rows
        for arm in ARMS:
            shifts.append(dict(arm=arm,split_regime=regime,contrast='patience30_minus_patience15',
                **compare(a[(a.arm==arm)&(a.split_regime==regime)],b[(b.arm==arm)&(b.split_regime==regime)],regime)))
    pd.DataFrame(shifts).to_csv(OUT/'patience_shifts.csv',index=False)
    # Surface every original full-budget MLP epoch, including untested peptide/locus cells.
    fit=pd.read_csv(BASE/'fit_compute.csv')
    epochs=fit[(fit.fraction==1)&fit.arm.isin(ARMS)][['arm','split_regime','fold','n_train','best_epoch']].rename(columns={'best_epoch':'best_epoch_patience15'})
    epochs=epochs.merge(scores[['arm','split_regime','fold','best_epoch']],on=['arm','split_regime','fold'],how='left',validate='one_to_one').rename(columns={'best_epoch':'best_epoch_patience30'})
    epochs.to_csv(OUT/'best_epochs.csv',index=False)
    lines=['# Post-hoc patience sensitivity', '',
        'Triggered by selected epochs ≤2 in the original allele-regime ESM-2 MLP fits. Only patience changes (15 → 30); the 150-epoch cap, allele-grouped validation, features, optimizer, widths, and fold seeds stay fixed. All three MLP approaches are rerun. The main benchmark and headline tables remain at patience 15.', '',
        '| Regime | Approach | Macro, patience 15 | Macro, patience 30 | Selected epoch ≤2 at patience 30 |', '|---|---|---:|---:|---:|']
    for regime in REGIMES:
        for arm in ARMS:
            a=old_scores[(old_scores.arm==arm)&(old_scores.split_regime==regime)]
            b=scores[(scores.arm==arm)&(scores.split_regime==regime)]
            lines.append(f'| {regime} | {arm} | {a[PRIMARY].mean():.3f} | {b[PRIMARY].mean():.3f} | {int((b.best_epoch<=2).sum())}/{len(b)} |')
    lines+=['', '| Regime | Approach − BLOSUM MLP | Patience 15 Δ [95% CI] | Patience 30 Δ [95% CI] |', '|---|---|---:|---:|']
    for regime in REGIMES:
        for arm in ARMS[1:]:
            vals=[]
            for p in [15,30]:
                c=contrasts[(contrasts.arm==arm)&(contrasts.split_regime==regime)&(contrasts.patience==p)].iloc[0]
                vals.append(f'{c.delta:+.3f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}]')
            lines.append(f'| {regime} | {arm} | '+' | '.join(vals)+' |')
    early=scores[scores.best_epoch<=2]
    lines += ['', f'**Early selected checkpoints persist in {len(early)}/18 fits at patience 30.**' if len(early) else '**No selected checkpoint remains at epoch ≤2 with patience 30.**', '',
        'An epoch-1 checkpoint has already received a full epoch of gradient updates; it is not an untrained model. The selected epoch does not equal the number of epochs attempted. Persistent early selection means this patience increase did not resolve the pattern, not that an underlying cause has been identified. No patience-60 run or alternative validation grouping is performed.', '',
        'Allele intervals pair five folds; strict intervals bootstrap paired eligible alleles within one fixed split. These nominal intervals do not account fully for overlapping training data, related alleles, multiplicity, or this post-hoc choice. Peptide folds were not rerun at patience 30, so their patience sensitivity is untested.', '',
        '## Full-budget stopping epochs', '', '| Approach | Regime | Fold | Training rows | Best epoch, patience 15 | Best epoch, patience 30 |', '|---|---|---:|---:|---:|---:|']
    for _,r in epochs.sort_values(['split_regime','arm','fold']).iterrows():
        new='not run' if pd.isna(r.best_epoch_patience30) else str(int(r.best_epoch_patience30))
        lines.append(f'| {r.arm} | {r.split_regime} | {int(r.fold)} | {int(r.n_train)} | {int(r.best_epoch_patience15)} | {new} |')
    lines+=['', 'Machine-readable records: [fold scores](../results/benchmark/patience_sensitivity/per_fold.csv), [paired comparisons](../results/benchmark/patience_sensitivity/paired_comparisons.csv), [within-approach changes](../results/benchmark/patience_sensitivity/patience_shifts.csv), [epochs](../results/benchmark/patience_sensitivity/best_epochs.csv), and [acceptance checks](../results/benchmark/patience_sensitivity/checks.json).']
    (REPORTS/'patience_sensitivity.md').write_text('\n'.join(lines)+'\n')


def main():
    # Record protected output hashes before preparation or any fit.
    design=lock_design()
    df=prepare()
    (OUT/'jobs').mkdir(exist_ok=True)
    rows=[];scores=[]
    for arm in ARMS:
        X=features(arm,df)
        for regime in REGIMES:
            for fold,train,test in split_indices(df,regime):
                path=OUT/'jobs'/f'{arm}_{regime}_{fold}.csv'
                meta=path.with_suffix('.json')
                original=json.loads((BASE/'jobs'/f'{arm}_{regime}_{fold}_1.json').read_text())
                train_sha=hashlib.sha256(df.index.to_numpy()[train].astype('<i8').tobytes()).hexdigest()
                assert original['training_row_ids_sha256']==train_sha
                if path.exists() and meta.exists():
                    r=pd.read_csv(path);m=json.loads(meta.read_text())
                    assert m['patience']==PATIENCE and m['training_row_ids_sha256']==train_sha
                else:
                    print(f'RUN {arm} {regime} fold={fold} patience={PATIENCE}',flush=True)
                    start=time.perf_counter()
                    xt,xv=np.asarray(X[train]),np.asarray(X[test])
                    with threadpool_limits(limits=4):
                        model=fit_baseline_nn(xt,df.y.iloc[train],seed=fold,groups=df.iloc[train].allele.to_numpy(),patience=PATIENCE)
                        pred=model.predict(xv)
                    r=pd.DataFrame(dict(row_index=df.index[test],allele=df.allele.iloc[test].to_numpy(),arm=arm,
                        split_regime=regime,fold=fold,patience=PATIENCE,y_true=df.y.iloc[test].to_numpy(),y_pred=pred))
                    m=dict(arm=arm,split_regime=regime,fold=fold,patience=PATIENCE,n_train=len(train),seed=fold,
                        training_row_ids_sha256=train_sha,best_epoch=model.best_epoch_,seconds=time.perf_counter()-start,
                        **metrics(r.y_true,r.y_pred,r.allele))
                    r.to_csv(path,index=False);meta.write_text(json.dumps(m,indent=2)+'\n')
                    print(f'{arm} {regime} f{fold} best_epoch={model.best_epoch_} macro={m[PRIMARY]:.4f}',flush=True)
                    del xt,xv,model
                assert m['n_train']==original['n_train']
                baseline=pd.read_csv(BASE/'jobs'/f'{arm}_{regime}_{fold}_1.csv')
                np.testing.assert_array_equal(r.row_index,baseline.row_index)
                # The original labels have made a CSV round trip; allow its rounding only.
                np.testing.assert_allclose(r.y_true,baseline.y_true,rtol=0,atol=1e-14)
                assert np.isfinite(r.y_pred).all()
                assert m['macro_eligible_alleles']==metrics(baseline.y_true,baseline.y_pred,baseline.allele)['macro_eligible_alleles']
                rows.append(r);scores.append(m)
        del X
    rows=pd.concat(rows,ignore_index=True);scores=pd.DataFrame(scores)
    assert len(scores)==18
    rows.to_csv(OUT/'per_row.csv',index=False);scores.to_csv(OUT/'per_fold.csv',index=False)
    report(rows,scores)
    assert protected_hashes()==design['protected_sha256'], 'Protected benchmark changed during sensitivity run'
    early=scores[scores.best_epoch<=2][['arm','split_regime','fold','best_epoch']].to_dict('records')
    checks=dict(passed=True,fits=18,prediction_rows=len(rows),original_benchmark_unchanged=True,
        training_hashes_match=True,test_rows_match=True,macro_eligibility_matches=True,
        remaining_best_epoch_le2=early,new_gpu_extraction=False,baseline_patience=15,patience=30,
        status='Completed one post-hoc sensitivity; no further patience search')
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
