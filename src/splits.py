"""Versioned outer partitions; all written and checked before new fits."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .config import DATA, RESULTS
from .data import main_data

OUT = RESULTS / 'splits'


def write_locked(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == content, f'Refuse to change locked design: {path}'
    else:
        path.write_text(json.dumps(content, indent=2) + '\n')


def _write_assignments(path, frame):
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), frame)
    else:
        frame.to_csv(path,index=False)
    return frame


def _make(df, group, k, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    labels=np.full(len(df),-1)
    for fold,(train,test) in enumerate(GroupKFold(k,shuffle=True,random_state=seed).split(df,groups=df[group])):
        assert not set(df.iloc[train][group]) & set(df.iloc[test][group])
        labels[test]=fold
    return _write_assignments(OUT/f'{group}_folds.csv',pd.DataFrame({'row_index':df.index,'fold':labels}))


def make_peptide_folds(df,k=5,seed=0):
    return _make(df,'peptide',k,seed)


def make_allele_folds(df,k=5,seed=0):
    return _make(df,'allele',k,seed)


def make_locus_holdout(df):
    OUT.mkdir(parents=True,exist_ok=True)
    loci=sorted(df.locus.unique())
    assert len(loci)>=2
    labels=df.locus.map({l:i for i,l in enumerate(loci)}).to_numpy()
    for fold in range(len(loci)):
        assert not set(df.loc[labels==fold,'locus']) & set(df.loc[labels!=fold,'locus'])
    return _write_assignments(OUT/'locus_folds.csv',pd.DataFrame({'row_index':df.index,'fold':labels}))


def make_strict_fold0(df,folds):
    assignment=folds.set_index('row_index').loc[df.index,'fold'].to_numpy()
    test=assignment==0
    shared=df.peptide.isin(df.loc[test,'peptide']).to_numpy()
    # 0=test; 1=train; -1=excluded training row. Only fold 0 is evaluated.
    labels=np.where(test,0,np.where(shared,-1,1))
    assert not set(df.loc[labels==0,'allele']) & set(df.loc[labels==1,'allele'])
    assert not set(df.loc[labels==0,'peptide']) & set(df.loc[labels==1,'peptide'])
    result=_write_assignments(OUT/'allele_folds_strict_fold0.csv',pd.DataFrame({'row_index':df.index,'fold':labels}))
    delta=dict(fold=0,test_rows=int(test.sum()),permissive_train_rows=int((~test).sum()),
               strict_train_rows=int((labels==1).sum()),removed_train_rows=int((labels==-1).sum()),
               encoding={'0':'test','1':'train','-1':'excluded shared-peptide training row'})
    write_locked(OUT/'strict_fold0_design.json',delta)
    pd.DataFrame([{k:v for k,v in delta.items() if k!='encoding'}]).to_csv(OUT/'strict_fold0_counts.csv',index=False)
    return result


def split_indices(df,regime):
    filename='allele_folds_strict_fold0.csv' if regime=='allele_strict' else f'{regime}_folds.csv'
    labels=pd.read_csv(OUT/filename).set_index('row_index').loc[df.index,'fold'].to_numpy()
    for fold in ([0] if regime=='allele_strict' else sorted(np.unique(labels))):
        train=np.where(labels==1)[0] if regime=='allele_strict' else np.where(labels!=fold)[0]
        test=np.where(labels==fold)[0]
        assert len(train) and len(test)
        yield int(fold),train,test


def nearest_train_allele_identity(df,folds,regime='allele'):
    labels=folds.set_index('row_index').loc[df.index,'fold'].to_numpy()
    rows=[]
    for fold in ([0] if regime=='allele_strict' else sorted(np.unique(labels))):
        train=df.loc[labels==1] if regime=='allele_strict' else df.loc[labels!=fold]
        test=df.loc[labels==fold]
        counts=train.groupby('allele').size()
        seqs=train.drop_duplicates('allele').set_index('allele').hla_pseudoseq
        for allele,g in test.groupby('allele'):
            seq=g.hla_pseudoseq.iloc[0]
            identities=seqs.map(lambda s:sum(a==b for a,b in zip(s,seq))/34)
            nearest=identities.sort_values(ascending=False,kind='stable').index[0]
            rows.append(dict(allele=allele,fold=int(fold),split_regime=regime,locus=g.locus.iloc[0],
                n_train_rows=int(counts.get(allele,0)),n_test_rows=len(g),
                zero_count=int(g.thalf_hours.eq(0).sum()),zero_fraction=float(g.thalf_hours.eq(0).mean()),
                n_distinct_nonzero=int(g.loc[g.thalf_hours>0,'thalf_hours'].nunique()),
                n_fold_train_rows=len(train),nearest_train_allele=nearest,
                nearest_train_allele_rows=int(counts[nearest]),max_identity_to_train=float(identities[nearest]),
                max_identity_to_train_allele=float(identities[nearest])))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/f'{regime}_distance.csv',index=False)
    return result


def prepare():
    df=main_data()
    OUT.mkdir(parents=True,exist_ok=True)
    design=dict(version=2,k=5,seed=0,splitter='GroupKFold(shuffle=True)',groups=['peptide','allele','locus'],
        exclusion='allele contains literal (',zeros='retained',
        locus_mapping={str(i):l for i,l in enumerate(sorted(df.locus.unique()))},
        absent_loci=sorted(set('ABC')-set(df.locus)),
        strict='Only allele fold 0; remove all training rows with a test peptide; 0=test,1=train,-1=removed',
        prior_design='results/archive/splits_v1/design.json; peptide/allele assignments unchanged',
        dataset_sha256=json.loads((DATA/'manifests/rasmussen_manifest.json').read_text())['sha256'])
    write_locked(OUT/'design.json',design)
    df.to_csv(DATA/'processed/rasmussen_main.csv')
    make_peptide_folds(df)
    allele=make_allele_folds(df)
    locus=make_locus_holdout(df)
    strict=make_strict_fold0(df,allele)
    for regime,folds in [('allele',allele),('locus',locus),('allele_strict',strict)]:
        nearest_train_allele_identity(df,folds,regime)
    stats=[]
    for a,g in df.groupby('allele'):
        stats.append(dict(allele=a,locus=g.locus.iloc[0],n_rows=len(g),zero_count=int(g.thalf_hours.eq(0).sum()),
            zero_fraction=float(g.thalf_hours.eq(0).mean()),n_distinct_nonzero=g.loc[g.thalf_hours>0,'thalf_hours'].nunique()))
    pd.DataFrame(stats).to_csv(DATA/'processed/rasmussen_allele_summary.csv',index=False)
    return df


if __name__=='__main__':
    print(f'Prepared {len(prepare())} rows; three regimes and strict fold 0 locked on disk.')
