from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src import workspace as w
from notes.website.webapp import create_app


def test_sequences_and_batch_validation():
    assert w.normalize_sequence('>example\nacdef\n ghik', protein=True)=='ACDEFGHIK'
    for sequence in ['ACDXFGHIK','ACDEFGHIK*','>a\nACDEFGHIK\n>b\nACDEFGHIK','A'*1001]:
        with pytest.raises(ValueError): w.normalize_sequence(sequence,protein=True)
    assert w.parse_batch('\ufeffpeptide,allele\n"fsvqrnlpf","HLA-B*15:01"\n')==[('FSVQRNLPF','HLA-B*15:01')]
    for text in ['peptide,allele\n','FSVQRNLPF,unknown','FSVQRNLPF,HLA-B*15:01,extra']:
        with pytest.raises(ValueError): w.parse_batch(text)
    assert [w.tier(x) for x in [0,1.999,2,5.999,6]]==['low','low','intermediate','intermediate','high']


def test_scan_preserves_positions_and_alleles_but_deduplicates_inference(monkeypatch):
    requests=[]
    def fake_score(pairs,arms):
        requests.extend(pairs)
        return [dict(peptide=p,allele=a,models=[dict(hours=1.)]) for p,a in pairs]
    monkeypatch.setattr(w,'score_pairs',fake_score)
    result=w.scan('A'*12,['HLA-A*02:01','HLA-B*15:01'])
    assert result['window_count']==4 and len(result['rows'])==8
    assert len(requests)==2 and len(set(requests))==2
    assert [r['position'] for r in result['rows']]==[1,2,3,4]*2
    with pytest.raises(ValueError): w.scan('A'*12,[])
    with pytest.raises(ValueError): w.scan('A'*12,['A'],'esm2_joint')


def test_new_pairing_uses_known_peptide_holdout_not_average(monkeypatch,tmp_path):
    monkeypatch.setattr(w,'RESULTS',tmp_path)
    (tmp_path/'models').mkdir()
    for fold in range(5): (tmp_path/'models'/f'blosum_nn_peptide_{fold}.joblib').touch()
    data=dict(df=pd.DataFrame(), seqs={'A':'A'*34,'B':'C'*34},
              peptide_folds={'AAAAAAAAA':4},pairs={('AAAAAAAAA','A'):0})
    monkeypatch.setattr(w,'dataset',lambda:data)
    monkeypatch.setattr(w,'model',lambda arm,fold:dict(model=SimpleNamespace(predict=lambda X:np.full(len(X),fold))))
    monkeypatch.setattr(w,'training_counts',lambda arm,fold:{'A':10,'B':20})
    result=w.score_pairs([('AAAAAAAAA','B'),('CCCCCCCCC','B')],['blosum_nn'])
    assert not result[0]['known_pair']
    assert result[0]['prediction_source']=='held_out_peptide_fold'
    assert result[0]['models'][0]['hours']==np.expm1(4)
    assert result[1]['prediction_source']=='five_model_mean'
    assert result[1]['models'][0]['hours']==np.expm1(2)


def test_known_predictions_match_saved_held_out_benchmark():
    result=w.predict_pair('FSVQRNLPF','HLA-B*15:01',include_plm=False)
    df=w.dataset()['df']
    idx=df[(df.peptide=='FSVQRNLPF')&(df.allele=='HLA-B*15:01')].index[0]
    rows=pd.read_csv(w.RESULTS/'benchmark/per_row.csv')
    rows=rows[(rows.row_index==idx)&(rows.split_regime=='peptide')].set_index('arm')
    for m in result['models']:
        np.testing.assert_allclose(m['hours'],np.expm1(max(0,rows.loc[m['id'],'y_pred'])),rtol=1e-5)
        assert m['low']<=m['hours']<=m['high']
    assert result['context']['measurements']==1070


def test_http_routes_validation_and_no_partial_invalid_batch(monkeypatch):
    client=TestClient(create_app())
    assert client.get('/').status_code==200
    assert 'Scan a protein' in client.get('/').text
    assert client.get('/assets/workspace.js').status_code==200
    assert len(client.get('/api/bootstrap').json()['alleles'])==72
    assert client.post('/api/scan',json={'sequence':'INVALID!','alleles':['HLA-A*02:01']}).status_code==400
    assert client.post('/api/scan',json={'sequence':'A'*10,'alleles':[]}).status_code==422
    def should_not_run(*args):
        raise AssertionError('Invalid batch should not reach inference')
    monkeypatch.setattr(w,'score_pairs',should_not_run)
    response=client.post('/api/batch',json={'text':'FSVQRNLPF,HLA-B*15:01\nBAD,HLA-B*15:01'})
    assert response.status_code==400 and 'Row 2' in response.json()['detail']
