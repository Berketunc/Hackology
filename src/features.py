import json
import numpy as np
from Bio.Align import substitution_matrices
from sklearn.preprocessing import OneHotEncoder
from .config import CHECKPOINT, HIDDEN_SIZE, DATA

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
BLOSUM = substitution_matrices.load('BLOSUM62')


def blosum_encode(seq):
    return np.asarray([[BLOSUM[a, b] for b in AMINO_ACIDS] for a in seq], dtype=np.float32).ravel()


def blosum_pairs(df):
    return np.stack([np.concatenate([blosum_encode(p), blosum_encode(h)])
                     for p, h in zip(df.peptide, df.hla_pseudoseq)])


def onehot_allele(df, encoder=None):
    """Pass the training-fitted encoder for held-out rows; unknown alleles map to zero."""
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        return encoder.fit_transform(df[['allele']]), encoder
    return encoder.transform(df[['allele']]), encoder


def validate_legacy_cache(path=None):
    path = path or DATA / 'processed'
    meta = json.loads((path / 'embeddings_meta.json').read_text())
    assert meta['checkpoint'] == CHECKPOINT and meta['hidden_size'] == HIDDEN_SIZE
    with np.load(path / 'embeddings_esm2.npz', allow_pickle=False) as cache:
        assert cache['peptide_emb'].shape[1] == HIDDEN_SIZE
        assert cache['mhc_emb'].shape[1] == HIDDEN_SIZE
    return meta


def load_embedding_arm(arm, df):
    root = DATA / 'processed'
    meta = json.loads((root / f'{arm}_meta.json').read_text())
    assert meta['checkpoint'] == CHECKPOINT and meta['hidden_size'] == HIDDEN_SIZE
    manifest = json.loads((DATA / 'manifests/rasmussen_manifest.json').read_text())
    assert meta['dataset_sha256'] == manifest['sha256']
    ids = np.load(root / f'{arm}_row_ids.npy', allow_pickle=False)
    assert np.array_equal(ids, df.index.to_numpy())
    X = np.load(root / f'{arm}.npy', mmap_mode='r', allow_pickle=False)
    assert X.shape == (len(df), meta['feature_dimension'])
    assert X.shape[1] == (2 * HIDDEN_SIZE if arm == 'esm2_mean' else 9 * HIDDEN_SIZE)
    return X


def embedding_arm_available(arm, root=None):
    """Metadata alone is committed; binaries must also exist to reuse a cache."""
    root = root or DATA / 'processed'
    return all((root / f'{arm}{suffix}').exists()
               for suffix in ['.npy', '_row_ids.npy', '_meta.json'])
