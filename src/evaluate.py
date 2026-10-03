"""Equal-weight within-allele ranking, explicit eligibility, and paired intervals."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, t
from sklearn.metrics import roc_auc_score

PRIMARY = 'macro_within_allele_spearman'
MIN_TEST_ROWS = 20
MIN_DISTINCT_NONZERO = 10
HIGH_ZERO_FRACTION = .5


def spearman(y_true,y_pred):
    if len(y_true)<3 or np.ptp(y_true)==0 or np.ptp(y_pred)==0:
        return float('nan')
    return float(spearmanr(y_true,y_pred).statistic)


def pooled_spearman(per_row):
    return spearman(per_row.y_true,per_row.y_pred)


def pearson(y_true,y_pred):
    if len(y_true)<3 or np.ptp(y_true)==0 or np.ptp(y_pred)==0:
        return float('nan')
    return float(pearsonr(y_true,y_pred).statistic)


def rmse_log1p(y_true,y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true)-y_pred)**2)))


def tier_auc(y_true_hours,y_pred,threshold):
    label=np.asarray(y_true_hours)>=threshold
    return float(roc_auc_score(label,y_pred)) if len(np.unique(label))==2 else float('nan')


def allele_metrics(per_row,allele_universe=None):
    """One row per allele, including every exclusion and undefined-score reason.

    Distinct positive log-labels have exactly the same cardinality as distinct hours.
    Threshold AUC comparisons remain in log space to avoid exp/log boundary rounding.
    """
    rows=[]
    groups=dict(tuple(per_row.groupby('allele',sort=True)))
    universe=sorted(groups) if allele_universe is None else sorted(set(allele_universe))
    for allele in universe:
        g=groups.get(allele,per_row.iloc[:0])
        nonzero=int(g.loc[g.y_true>0,'y_true'].nunique())
        reasons=[]
        if len(g)<MIN_TEST_ROWS: reasons.append('fewer_than_20_test_rows')
        if nonzero<MIN_DISTINCT_NONZERO: reasons.append('fewer_than_10_distinct_nonzero_half_lives')
        eligible=not reasons
        rho=spearman(g.y_true,g.y_pred)
        if eligible and not np.isfinite(rho): reasons.append('undefined_prediction_spearman')
        rows.append(dict(allele=allele,n_test_rows=len(g),n_distinct_nonzero=nonzero,
            zero_count=int(g.y_true.eq(0).sum()),zero_fraction=float(g.y_true.eq(0).mean()),
            high_zero_fraction=bool(g.y_true.eq(0).mean()>=HIGH_ZERO_FRACTION),
            label_eligible=eligible,include_in_macro=eligible and np.isfinite(rho),
            exclusion_reason=';'.join(reasons),spearman=rho,
            auc_2h=tier_auc(g.y_true,g.y_pred,np.log1p(2)),
            auc_6h=tier_auc(g.y_true,g.y_pred,np.log1p(6))))
    return pd.DataFrame(rows)


def macro_within_allele_spearman(per_row):
    stats=allele_metrics(per_row)
    eligible=stats[stats.label_eligible]
    # Undefined predictions cannot improve an arm's macro by silently dropping alleles.
    if eligible.empty or not eligible.include_in_macro.all(): return float('nan')
    return float(eligible.spearman.mean())


def metrics(y_true,y_pred,alleles=None,allele_universe=None):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    result=dict(pooled_spearman=spearman(y_true,y_pred),pearson=pearson(y_true,y_pred),
        rmse_log1p=rmse_log1p(y_true,y_pred),auc_2h=tier_auc(y_true,y_pred,np.log1p(2)),
        auc_6h=tier_auc(y_true,y_pred,np.log1p(6)))
    if alleles is not None:
        frame=pd.DataFrame(dict(y_true=y_true,y_pred=y_pred,allele=np.asarray(alleles)))
        stats=allele_metrics(frame,allele_universe)
        eligible=stats[stats.label_eligible]
        macro=float(eligible.spearman.mean()) if len(eligible) and eligible.include_in_macro.all() else float('nan')
        result.update({PRIMARY:macro,'pooled_minus_macro':result['pooled_spearman']-macro,
            'macro_eligible_alleles':int(stats.label_eligible.sum()),
            'macro_excluded_alleles':int((~stats.include_in_macro).sum())})
    return result


def paired_delta_ci(arm,baseline,by='fold',value='value',method='t',seed=0,n_boot=10000):
    """Paired 95% CI on aligned unit scores. t intervals are descriptive for CV folds.

    Bootstrap resamples whole paired units, never independent rows from each model.
    For one fixed split use allele units and label the interval conditional on that split.
    """
    assert not arm[by].duplicated().any() and not baseline[by].duplicated().any()
    joined=arm[[by,value]].merge(baseline[[by,value]],on=by,how='outer',validate='one_to_one',suffixes=('_arm','_baseline'),indicator=True)
    assert joined['_merge'].eq('both').all(), 'Paired units must match exactly'
    good=np.isfinite(joined[value+'_arm']) & np.isfinite(joined[value+'_baseline'])
    delta=(joined.loc[good,value+'_arm']-joined.loc[good,value+'_baseline']).to_numpy()
    n=len(delta)
    mean=float(delta.mean()) if n else float('nan')
    low=high=float('nan')
    if n>=2:
        if method=='t':
            half=float(t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n))
            low,high=mean-half,mean+half
        elif method=='bootstrap':
            rng=np.random.default_rng(seed)
            boot=delta[rng.integers(0,n,size=(n_boot,n))].mean(axis=1)
            low,high=map(float,np.quantile(boot,[.025,.975]))
        else: raise ValueError(method)
    return dict(delta=mean,ci_low=low,ci_high=high,n_pairs=n,n_unscorable_pairs=int((~good).sum()),
                pairing_unit=by,ci_method=method,confidence=.95)


def stratify_by_allele_distance(per_row_results,allele_distance):
    keys=['allele','fold']
    if 'split_regime' in per_row_results and 'split_regime' in allele_distance: keys+=['split_regime']
    frame=per_row_results.merge(allele_distance,on=keys,validate='many_to_one')
    frame['identity_band']=pd.cut(frame.max_identity_to_train,[0,.8,.9,.95,1.0001],right=False,
        labels=['<80%','80–90%','90–95%','95–100%'])
    frame['n_train_rows_band']=pd.cut(frame.n_train_rows,[-1,0,100,500,1000,np.inf],
        labels=['0','1–100','101–500','501–1000','>1000'])
    frame['nearest_support_band']=pd.cut(frame.nearest_train_allele_rows,[0,100,500,1000,np.inf],
        labels=['1–100','101–500','501–1000','>1000'])
    rows=[]
    for stratifier in ['identity_band','n_train_rows_band','nearest_support_band']:
        for (arm,band),g in frame.groupby(['arm',stratifier],observed=True):
            rows.append(dict(arm=arm,stratifier=stratifier,band=str(band),n_rows=len(g),
                **metrics(g.y_true,g.y_pred,g.allele)))
    return pd.DataFrame(rows)
