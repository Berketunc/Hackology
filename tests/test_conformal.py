import numpy as np
import pandas as pd
import pytest

from src.conformal import conformal_cutoff, hour_interval, calibration_table, summarize


def test_finite_sample_quantile_and_small_sample_abstention():
    assert conformal_cutoff(np.arange(1, 20)) == (19., 19)
    assert conformal_cutoff(np.arange(1, 101)) == (96., 96)
    assert conformal_cutoff(np.arange(18)) == (float('inf'), 19)
    assert conformal_cutoff([]) == (float('inf'), 1)
    with pytest.raises(ValueError):
        conformal_cutoff([np.nan])


def test_hour_interval_uses_identical_clipping_and_handles_unbounded():
    center, low, high = hour_interval([-2., np.log1p(4)], [1., np.log(2)])
    np.testing.assert_allclose(center, [0., 4.])
    np.testing.assert_allclose(low, [0., 1.5])
    np.testing.assert_allclose(high, [np.e-1, 9.])
    assert hour_interval(-4., float('inf')) == (0., 0., float('inf'))


def test_allele_calibration_never_pools_shared_peptides_or_sparse_alleles():
    frames = [pd.DataFrame(dict(allele=a, peptide=[str(i) for i in range(n)],
                y_true=np.arange(1, n+1)*scale, y_pred=0.))
              for a, n, scale in [('A', 20, 1.), ('B', 20, 10.), ('C', 3, 1.)]]
    rows = pd.concat(frames)
    stats = calibration_table(rows, {'A':100, 'B':100, 'C':10}, ['A','B','C','D']).set_index('allele')
    assert stats.loc['A','q_log'] == 20.
    assert stats.loc['B','q_log'] == 200.
    assert stats.loc['A','n_calibration'] == 20
    assert stats.loc['C','status'] == 'insufficient_calibration'
    assert stats.loc['D','status'] == 'unseen_allele'
    assert np.isinf(stats.loc['C','q_log']) and np.isinf(stats.loc['D','q_log'])
    with pytest.raises(AssertionError):
        calibration_table(pd.concat([rows, rows.iloc[:1]]), {'A':100}, ['A'])


def test_unbounded_intervals_do_not_inflate_headline_coverage():
    rows = pd.DataFrame(dict(status=['available','available','insufficient_calibration'],
                             covered=[True, False, True], width_hours=[2.,4.,np.inf]))
    result = summarize(rows)
    assert result['coverage_finite'] == .5
    assert result['n_without_finite'] == 1
    assert result['mean_width_hours_finite'] == 3.
    assert result['coverage_including_unbounded'] == 2/3


def test_demo_routes_new_pairing_by_peptide_and_novel_peptide_to_fixed_fold(tmp_path, monkeypatch):
    import src.conformal as module
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'RESULTS', tmp_path)
    monkeypatch.setattr(module, 'OUT', tmp_path)
    monkeypatch.setattr(module, 'MODEL_DIR', tmp_path)
    (tmp_path/'splits').mkdir()
    pd.DataFrame(dict(row_index=[0,1,2], fold=[2,0,1])).to_csv(tmp_path/'splits/peptide_folds.csv', index=False)
    for fold in [0,2]:
        (tmp_path/f'fold{fold}.joblib').touch()
    pd.DataFrame(dict(scope=['fold','fold'],fold=[0,2],coverage_finite=[.95,.96],n_finite=[100,100])).to_csv(tmp_path/'coverage.csv', index=False)
    pd.DataFrame(dict(allele=['B'],coverage_finite=[.95],n_finite=[100])).to_csv(tmp_path/'coverage_by_allele.csv', index=False)
    df=pd.DataFrame(dict(peptide=['AAAAAAAAA','CCCCCCCCC','DDDDDDDDD'],allele=['A','B','B'],hla_pseudoseq=['A'*34]*3))
    selected=[]
    def bundle(fold):
        selected.append(fold)
        return dict(model=SimpleNamespace(predict=lambda x: np.array([1.])),fit_row_ids=[1],calibration_row_ids=[2],
                    calibration=pd.DataFrame(dict(allele=['B'],q_log=[1.],n_calibration=[50],n_train_rows=[100],status=['available'])))
    monkeypatch.setattr(module,'load_bundle',bundle)
    result=module.predict_interval('AAAAAAAAA','B',df)
    assert selected[-1]==2 and result['fold']==2  # peptide exists, exact pair does not
    module.predict_interval('EEEEEEEEE','B',df)
    assert selected[-1]==0
    with pytest.raises(AssertionError):  # prevent displaying a calibrated range for a fitted peptide
        module.predict_interval('CCCCCCCCC','B',df)
