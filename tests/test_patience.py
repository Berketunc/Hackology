"""Changing the patience interface must preserve the original default fit."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.config import RESULTS
from src.data import main_data
from src.features import blosum_pairs
from src.models import fit_baseline_nn
from src.splits import split_indices


def test_default_patience_preserves_reference_fit():
    df=main_data()
    fold,train,test=next(split_indices(df,'allele'))
    features=blosum_pairs(df)
    with threadpool_limits(limits=4):
        model=fit_baseline_nn(features[train],df.y.iloc[train],seed=fold,
                              groups=df.allele.iloc[train].to_numpy())
        predictions=model.predict(features[test])
    path=RESULTS/'benchmark/jobs/blosum_nn_allele_0_1.csv'
    baseline=pd.read_csv(path)
    metadata=json.loads(path.with_suffix('.json').read_text())
    np.testing.assert_array_equal(df.index[test],baseline.row_index)
    np.testing.assert_allclose(predictions,baseline.y_pred,rtol=1e-6,atol=1e-7)
    assert model.best_epoch_==metadata['best_epoch']
