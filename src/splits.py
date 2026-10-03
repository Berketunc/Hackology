"""Persist all outer partitions before fitting any model."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .config import DATA, RESULTS
from .data import main_data

OUT = RESULTS / 'splits'


def _make(df, group, k, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    design = dict(k=k, seed=seed, splitter='GroupKFold(shuffle=True)',
                  groups=['peptide', 'allele'], exclusion='allele contains literal (',
                  zeros='retained', dataset_sha256=json.loads((DATA / 'manifests/rasmussen_manifest.json').read_text())['sha256'])
    path = OUT / 'design.json'
    if path.exists():
        assert json.loads(path.read_text()) == design, 'Refuse to change locked design'
    else:
        path.write_text(json.dumps(design, indent=2) + '\n')
    labels = np.full(len(df), -1)
    for fold, (train, test) in enumerate(GroupKFold(k, shuffle=True, random_state=seed).split(df, groups=df[group])):
        assert not set(df.iloc[train][group]) & set(df.iloc[test][group])
        labels[test] = fold
    result = pd.DataFrame({'row_index': df.index, 'fold': labels})
    target = OUT / f'{group}_folds.csv'
    if target.exists():
        pd.testing.assert_frame_equal(pd.read_csv(target), result)
    else:
        result.to_csv(target, index=False)
    return result


def make_peptide_folds(df, k=5, seed=0):
    return _make(df, 'peptide', k, seed)


def make_allele_folds(df, k=5, seed=0):
    return _make(df, 'allele', k, seed)


def nearest_train_allele_identity(df, folds):
    joined = df.join(folds.set_index('row_index'))
    rows = []
    for fold in sorted(joined.fold.unique()):
        train, test = joined[joined.fold != fold], joined[joined.fold == fold]
        counts = train.groupby('allele').size()
        seqs = train.drop_duplicates('allele').set_index('allele').hla_pseudoseq
        for allele, group in test.groupby('allele'):
            seq = group.hla_pseudoseq.iloc[0]
            identities = seqs.map(lambda s: sum(a == b for a, b in zip(s, seq)) / 34)
            nearest = identities.sort_values(ascending=False, kind='stable').index[0]
            rows.append(dict(allele=allele, fold=fold, n_train_rows=int(counts.get(allele, 0)),
                             n_test_rows=len(group), n_fold_train_rows=len(train),
                             nearest_train_allele=nearest, nearest_train_allele_rows=int(counts[nearest]),
                             max_identity_to_train_allele=float(identities[nearest])))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'allele_distance.csv', index=False)
    return result


def prepare():
    df = main_data()
    (DATA / 'processed').mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / 'processed/rasmussen_main.csv')
    make_peptide_folds(df)
    folds = make_allele_folds(df)
    nearest_train_allele_identity(df, folds)
    return df


if __name__ == '__main__':
    print(f'Prepared {len(prepare())} rows and both locked split regimes.')
