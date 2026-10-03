"""Organiser data, validated without discarding reported zeros."""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .config import DATA

COLUMNS = ['allele', 'peptide', 'thalf_hours', 'hla_seq', 'hla_pseudoseq']
AA = set('ACDEFGHIKLMNPQRSTVWY')


def load_rasmussen(path=None):
    path = path or os.environ.get('RASMUSSEN_CSV', DATA / 'external/rasmussen_et_al_dataset.csv')
    from pathlib import Path
    path = Path(path)
    saved_manifest = DATA / 'manifests/rasmussen_manifest.json'
    fallback = not path.exists() and path.name == 'rasmussen_et_al_dataset.csv'
    if fallback:
        derivative = DATA / 'processed/rasmussen_all.csv'
        prior = json.loads(saved_manifest.read_text())
        assert hashlib.sha256(derivative.read_bytes()).hexdigest() == prior['processed_sha256']
        df = pd.read_csv(derivative, dtype=str)[COLUMNS]
        source_sha = prior['sha256']
    else:
        df = pd.read_csv(path, dtype=str)
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert list(df.columns) == COLUMNS, df.columns
    assert not df[COLUMNS].isna().any().any()
    df['thalf_hours'] = pd.to_numeric(df.thalf_hours, errors='raise')
    assert np.isfinite(df.thalf_hours).all() and (df.thalf_hours >= 0).all()
    for col, length in [('peptide', 9), ('hla_pseudoseq', 34), ('hla_seq', 182)]:
        assert df[col].str.len().eq(length).all(), col
        assert df[col].map(lambda x: set(x) <= AA).all(), col
    assert len(df) == 28166 and df.allele.nunique() == 75
    assert not df.duplicated(['allele', 'peptide']).any()
    assert df.groupby('allele')[['hla_seq', 'hla_pseudoseq']].nunique().eq(1).all().all()
    df.index.name = 'row_index'
    df['y'] = np.log1p(df.thalf_hours)
    df['is_engineered'] = df.allele.str.contains('(', regex=False)
    df['tier'] = pd.cut(df.thalf_hours, [-np.inf, 2, 6, np.inf], right=False,
                        labels=['low', 'intermediate', 'high'])
    manifest = dict(source_file=path.name, sha256=source_sha,
                    rows=len(df), alleles=df.allele.nunique(), peptides=df.peptide.nunique(),
                    zeros=int(df.thalf_hours.eq(0).sum()),
                    engineered_rows=int(df.is_engineered.sum()),
                    engineered_alleles=sorted(df.loc[df.is_engineered, 'allele'].unique()),
                    main_rows=int((~df.is_engineered).sum()),
                    main_zeros=int((df.thalf_hours.eq(0) & ~df.is_engineered).sum()),
                    zero_policy='Retained as exact recorded zero; possible left censoring.',
                    source_url='https://docs.google.com/spreadsheets/d/1NtZNvcF3u0KFn-1bbuA50CF3IXvs1l4HfbaR3KRLvso/edit')
    (DATA / 'manifests').mkdir(parents=True, exist_ok=True)
    derivative = DATA / 'processed/rasmussen_all.csv'
    derivative.parent.mkdir(parents=True, exist_ok=True)
    if not fallback:
        df.to_csv(derivative)
    manifest['processed_sha256'] = hashlib.sha256(derivative.read_bytes()).hexdigest()
    target = DATA / 'manifests/rasmussen_manifest.json'
    if target.exists():
        assert json.loads(target.read_text())['sha256'] == manifest['sha256'], 'Dataset changed'
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    return df


def main_data():
    df = load_rasmussen()
    return df.loc[~df.is_engineered].copy()
