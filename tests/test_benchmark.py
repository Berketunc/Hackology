import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data import main_data
from src.splits import prepare
from src.config import RESULTS
from src.features import blosum_encode, onehot_allele, validate_legacy_cache
from src.evaluate import metrics, tier_auc, spearman
from src.models import fit_ridge, fit_baseline_nn


def test_dataset_and_saved_group_partitions():
    df = prepare()
    assert len(df) == 27031
    assert (df.thalf_hours == 0).sum() == 4711
    assert df.allele.nunique() == 72
    assert not df.is_engineered.any()
    for regime in ['peptide', 'allele']:
        assignment = pd.read_csv(RESULTS / f'splits/{regime}_folds.csv').set_index('row_index')
        frame = df.join(assignment)
        assert frame.fold.notna().all()
        assert frame.groupby(regime).fold.nunique().eq(1).all()
        for fold in range(5):
            assert not set(frame.loc[frame.fold == fold,regime]) & set(frame.loc[frame.fold != fold,regime])
    distance = pd.read_csv(RESULTS / 'splits/allele_distance.csv')
    assert distance.n_train_rows.eq(0).all()
    assert distance.max_identity_to_train_allele.between(0, 1).all()


def test_unknown_allele_not_learned_from_test():
    train = pd.DataFrame({'allele':['A','B','A']})
    x, encoder = onehot_allele(train)
    test, _ = onehot_allele(pd.DataFrame({'allele':['UNSEEN']}), encoder)
    assert x.shape == (3,2) and not test.any()
    assert len(blosum_encode('ACDEFGHIK')) == 180


def test_metric_direction_ties_and_single_class():
    y = np.log1p([0,0,1,2,6,10])
    assert np.isclose(spearman(y,y),1)
    assert tier_auc(np.expm1(y), y, 2) == 1
    assert np.isnan(tier_auc([0,0], [1,2], 6))
    assert metrics(y,y)['rmse_log1p'] == 0
    z=np.log1p([0.,6.,7.])
    assert metrics(z, np.array([0.,2.,1.]))['auc_6h']==1.0


def test_training_preprocessing_and_legacy_cache():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60,8))
    y = x[:,0] + .2*x[:,1]
    model = fit_ridge(x, y)
    np.testing.assert_allclose(model.named_steps['scale'].mean_, x.mean(0))
    before = model.named_steps['scale'].mean_.copy()
    model.predict(np.full((4,8), 10000.))
    np.testing.assert_array_equal(before, model.named_steps['scale'].mean_)
    validate_legacy_cache()


def test_nn_inner_group_validation_scaler():
    from sklearn.model_selection import GroupShuffleSplit
    x = np.random.default_rng(0).normal(size=(80,6))
    groups = np.repeat(np.arange(20),4)
    train, val = next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=0).split(x,groups=groups))
    model = fit_baseline_nn(x,x[:,0],groups=groups,max_epochs=2)
    np.testing.assert_allclose(model.scaler.mean_,x[train].mean(0))
    assert not set(groups[train]) & set(groups[val])


def test_corrected_audit_regression(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from archive.iedb_pilot import check_stability_data as audit
    from src.config import ROOT
    DATA = ROOT / 'archive/iedb_pilot/data'
    # Reviewed derivatives preserve all original 10,605 retained records.
    frame=pd.concat([pd.read_csv(DATA/'processed/eligible_records.csv',dtype=str),
                     pd.read_csv(DATA/'processed/excluded_records.csv',dtype=str)],ignore_index=True).fillna('')
    assert len(frame)==10605
    monkeypatch.setattr(audit,'read_iedb',lambda _: (frame.copy(),5536251,0))
    result=audit.audit(SimpleNamespace(out=str(tmp_path/'audit'),input='retained-original-fields'))
    assert result['no_automatic_flags_not_yet_verified']==6101
    assert result['known_replicate_comment_positive_candidates']==5815
    assert result['comments_flagged']==6
    # A changed comment containing a bound must not inherit the exemption.
    idx=frame.index[frame.comments.str.contains('at least two independent',regex=False)][0]
    frame.loc[idx,'comments'] += ' Half-life at least 8 hours.'
    result=audit.audit(SimpleNamespace(out=str(tmp_path/'bounds'),input='retained-original-fields'))
    assert result['comments_flagged']==7



def test_clone_metadata_does_not_imply_embedding_cache(tmp_path):
    from src.features import embedding_arm_available
    (tmp_path/'esm2_mean_meta.json').write_text('{}')
    assert not embedding_arm_available('esm2_mean',tmp_path)
    np.save(tmp_path/'esm2_mean.npy',np.zeros((2,2560)))
    assert not embedding_arm_available('esm2_mean',tmp_path)
    np.save(tmp_path/'esm2_mean_row_ids.npy',np.array([0,1]))
    assert embedding_arm_available('esm2_mean',tmp_path)


def test_macro_eligibility_equal_weights_and_explicit_exclusions():
    from src.evaluate import macro_within_allele_spearman, allele_metrics
    frames=[]
    for allele,n,direction in [('large',100,1),('small',20,-1),('too_few_rows',19,1)]:
        y=np.log1p(np.arange(1,n+1))
        frames.append(pd.DataFrame({'allele':allele,'y_true':y,'y_pred':direction*y}))
    frames.append(pd.DataFrame({'allele':'too_few_values','y_true':np.log1p(np.tile(np.arange(10),3)),'y_pred':np.arange(30)}))
    rows=pd.concat(frames)
    assert abs(macro_within_allele_spearman(rows))<1e-12
    audit=allele_metrics(rows).set_index('allele')
    assert audit.include_in_macro.sum()==2
    assert '20_test_rows' in audit.loc['too_few_rows','exclusion_reason']
    assert '10_distinct_nonzero' in audit.loc['too_few_values','exclusion_reason']
    # Undefined predictions must not silently remove one of the eligible alleles.
    rows.loc[rows.allele=='small','y_pred']=0
    assert np.isnan(macro_within_allele_spearman(rows))


def test_primary_is_not_pooled_and_zero_rich_auc_remains_available():
    from src.evaluate import macro_within_allele_spearman, pooled_spearman, allele_metrics
    y=np.tile(np.arange(1,21),2)+np.repeat([0,100],20)
    pred=np.tile(np.arange(20,0,-1),2)+np.repeat([0,100],20)
    frame=pd.DataFrame({'allele':np.repeat(['A','B'],20),'y_true':np.log1p(y),'y_pred':pred})
    assert np.isclose(macro_within_allele_spearman(frame),-1)
    assert pooled_spearman(frame)>0
    y=np.log1p(np.r_[np.zeros(70),np.arange(1,31)])
    stats=allele_metrics(pd.DataFrame({'allele':'Z','y_true':y,'y_pred':y})).iloc[0]
    assert stats.include_in_macro and stats.high_zero_fraction
    assert stats.zero_fraction==.7 and stats.auc_2h==stats.auc_6h==1


def test_paired_ci_aligns_units_and_matches_scipy():
    import pytest
    from scipy.stats import ttest_rel
    from src.evaluate import paired_delta_ci
    a=pd.DataFrame({'fold':np.arange(5),'value':[.4,.6,.2,.5,.3]})
    b=pd.DataFrame({'fold':np.arange(5),'value':[.3,.2,.3,.4,.1]})
    expected=ttest_rel(a.value,b.value).confidence_interval()
    result=paired_delta_ci(a.sample(frac=1,random_state=4),b)
    assert np.isclose(result['ci_low'],expected.low) and np.isclose(result['ci_high'],expected.high)
    assert result['n_pairs']==5 and result['n_unscorable_pairs']==0
    assert paired_delta_ci(a,b,method='bootstrap')==paired_delta_ci(a,b,method='bootstrap')
    with pytest.raises(AssertionError): paired_delta_ci(a,b.iloc[:-1])
    assert np.isnan(paired_delta_ci(a.iloc[:1],b.iloc[:1])['ci_low'])


def test_locus_and_strict_splits_are_group_safe():
    from src.splits import split_indices
    df=prepare()
    assert set(df.locus)=={'A','B'}
    for fold,tr,te in split_indices(df,'locus'):
        assert not set(df.iloc[tr].locus)&set(df.iloc[te].locus)
        assert not set(df.iloc[tr].allele)&set(df.iloc[te].allele)
    _,tr,te=next(split_indices(df,'allele_strict'))
    _,permissive_train,permissive_test=next(split_indices(df,'allele'))
    np.testing.assert_array_equal(te,permissive_test)
    assert len(tr)==7794 and len(permissive_train)-len(tr)==15085
    assert not set(df.iloc[tr].peptide)&set(df.iloc[te].peptide)
    assert set(tr)<set(permissive_train)


def test_absent_test_alleles_have_explicit_zero_count_audit():
    from src.evaluate import allele_metrics,metrics
    y=np.log1p(np.arange(1,21))
    frame=pd.DataFrame({'allele':'A','y_true':y,'y_pred':y})
    stats=allele_metrics(frame,['A','B']).set_index('allele')
    assert stats.loc['B','n_test_rows']==0 and not stats.loc['B','include_in_macro']
    result=metrics(y,y,np.repeat('A',20),['A','B'])
    assert result['macro_eligible_alleles']==1 and result['macro_excluded_alleles']==1
