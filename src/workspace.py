"""Real model inference behind the protein-scanning workspace.

Bulk scans use the sequence baselines. ESM-2 is an explicit, potentially slower
comparison option. A known peptide always uses its held-out fold, even for a new
allele pairing; the five-fit range remains an uncalibrated sensitivity diagnostic.
"""
import csv
import io
import threading
from functools import lru_cache

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import RESULTS
from .data import AA, main_data
from .demo import model, embeddings, new_pair_features, overlap_lookup
from .features import blosum_encode, onehot_allele

MODEL_NAMES = {
    'blosum_nn': 'BLOSUM MLP (reference)',
    'blosum_ridge': 'BLOSUM Ridge',
    'onehot_ridge': 'Allele-ID Ridge (floor)',
    'esm2_mean': 'ESM-2 independent mean',
    'esm2_joint': 'ESM-2 joint sequence encoding',
}
BASELINES = list(MODEL_NAMES)[:3]
EXAMPLE_SEQUENCE = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV'
MAX_SEQUENCE = 1000
MAX_ALLELES = 6
MAX_BATCH = 200
INFERENCE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def dataset():
    df = main_data()
    folds = pd.read_csv(RESULTS/'splits/peptide_folds.csv').set_index('row_index').fold
    return dict(df=df, seqs=df.drop_duplicates('allele').set_index('allele').hla_pseudoseq.to_dict(),
                counts=df.groupby('allele').size().to_dict(),
                peptide_folds=dict(zip(df.peptide, folds.loc[df.index])),
                pairs={(r.peptide,r.allele):i for i,r in enumerate(df.itertuples())})


def normalize_sequence(sequence, protein=False):
    text = str(sequence).strip()
    lines = text.splitlines()
    if protein and any(line.startswith('>') for line in lines):
        if sum(line.startswith('>') for line in lines) != 1 or not lines[0].startswith('>'):
            raise ValueError('Paste one protein sequence or a single FASTA record.')
        text = ''.join(lines[1:])
    seq = ''.join(text.split()).upper()
    if not seq or set(seq)-AA:
        raise ValueError('Use only the 20 standard amino-acid letters; ambiguous residues and stop symbols are unsupported.')
    if protein and not 9 <= len(seq) <= MAX_SEQUENCE:
        raise ValueError(f'Protein length must be between 9 and {MAX_SEQUENCE:,} residues.')
    if not protein and len(seq) != 9:
        raise ValueError('Each peptide must contain exactly 9 amino-acid letters.')
    return seq


def validate_pair(peptide, allele):
    peptide = normalize_sequence(peptide)
    allele = str(allele).strip()
    if allele not in dataset()['seqs']:
        raise ValueError(f'Unknown allele: {allele}. Choose one of the dataset alleles.')
    return peptide, allele


def tier(hours):
    return 'high' if hours >= 6 else 'intermediate' if hours >= 2 else 'low'


@lru_cache(maxsize=150)
def training_counts(arm, fold):
    df = dataset()['df']
    return df.loc[model(arm, fold)['row_indices']].groupby('allele').size().to_dict()


def score_pairs(pairs, arms):
    """Vectorized frozen-model scoring; all returned values are actual predictions."""
    pairs = [validate_pair(p,a) for p,a in pairs]
    if not pairs:
        return []
    if not arms or any(a not in MODEL_NAMES for a in arms):
        raise ValueError('Choose an available model.')
    state = dataset()
    df = state['df']
    for arm in arms:
        if not all((RESULTS/'models'/f'{arm}_peptide_{f}.joblib').exists() for f in range(5)):
            raise RuntimeError(f'{MODEL_NAMES[arm]} is unavailable: its fitted models have not been generated.')
    frame = pd.DataFrame(pairs, columns=['peptide','allele'])
    result = [dict(peptide=p, allele=a, known_pair=(p,a) in state['pairs'],
                   prediction_source='held_out_peptide_fold' if p in state['peptide_folds'] else 'five_model_mean',
                   models=[]) for p,a in pairs]
    with INFERENCE_LOCK, threadpool_limits(limits=4):
        peptide_features = np.stack(frame.peptide.map(blosum_encode))
        hla_features = np.stack(frame.allele.map(lambda a: blosum_encode(state['seqs'][a])))
        blosum_features = np.column_stack([peptide_features, hla_features])
        novel_embeddings = {}
        for arm in arms:
            if arm.startswith('esm2'):
                cached = embeddings(arm)
                vectors = []
                for p,a in pairs:
                    pos = state['pairs'].get((p,a))
                    if pos is not None:
                        vectors.append(cached[pos])
                    else:
                        if (p,a) not in novel_embeddings:
                            novel_embeddings[p,a] = new_pair_features(p, state['seqs'][a])
                        vectors.append(novel_embeddings[p,a][arm][0])
                X = np.stack(vectors)
            else:
                X = blosum_features
            predictions = []
            for fold in range(5):
                bundle = model(arm, fold)
                if arm == 'onehot_ridge':
                    encoded,_ = onehot_allele(frame, bundle['encoder'])
                    X = np.column_stack([peptide_features, encoded])
                predictions.append(bundle['model'].predict(X))
            pred = np.asarray(predictions, dtype=float)
            if not np.isfinite(pred).all():
                raise RuntimeError('The model returned a non-finite prediction; no result was recorded.')
            for i,(p,a) in enumerate(pairs):
                fold = state['peptide_folds'].get(p)
                center = pred[fold,i] if fold is not None else pred[:,i].mean()
                hours, low, high = np.expm1(np.maximum(0., [center, pred[:,i].min(), pred[:,i].max()]))
                if not np.isfinite([hours,low,high]).all():
                    raise RuntimeError('The model prediction exceeds the supported numerical range.')
                count = training_counts(arm,fold).get(a,0) if fold is not None else round(np.mean([training_counts(arm,f).get(a,0) for f in range(5)]))
                result[i]['models'].append(dict(id=arm, name=MODEL_NAMES[arm], hours=float(hours),
                    low=float(low), high=float(high), tier=tier(hours), training_rows=int(count)))
    return result


def pair_context(peptide, allele):
    state = dataset()
    df = state['df']
    rows = df[df.allele==allele]
    candidates = [a for a,n in state['counts'].items() if n > state['counts'][allele]]
    nearest = None
    if candidates:
        identity = lambda a: sum(x==y for x,y in zip(state['seqs'][a],state['seqs'][allele]))/34
        best = max(sorted(candidates), key=identity)
        nearest = dict(allele=best, measurements=state['counts'][best], identity=identity(best))
    pair = df[(df.peptide==peptide)&(df.allele==allele)]
    return dict(measurements=len(rows), zero_fraction=float(rows.thalf_hours.eq(0).mean()),
                nearest=nearest, recorded_hours=None if pair.empty else float(pair.thalf_hours.iloc[0]),
                exposure=overlap_lookup(peptide,allele))


def predict_pair(peptide, allele, include_plm=True):
    p,a = validate_pair(peptide,allele)
    result = score_pairs([(p,a)], list(MODEL_NAMES) if include_plm else BASELINES)[0]
    result['context'] = pair_context(p,a)
    return result


def scan(sequence, alleles, arm='blosum_nn'):
    sequence = normalize_sequence(sequence, protein=True)
    alleles = list(dict.fromkeys(alleles))
    if not 1 <= len(alleles) <= MAX_ALLELES:
        raise ValueError(f'Select between 1 and {MAX_ALLELES} alleles.')
    if arm not in BASELINES:
        raise ValueError('Protein scans use sequence baselines; compare selected pairs with ESM-2 separately.')
    windows = [(i+1, sequence[i:i+9]) for i in range(len(sequence)-8)]
    # Repeated peptide windows keep their positions, but inference is deduplicated.
    unique = list(dict.fromkeys((p,a) for a in alleles for _,p in windows))
    scored = score_pairs(unique, [arm])
    lookup = {(r['peptide'],r['allele']):r for r in scored}
    results = [dict(position=pos, **lookup[p,a]) for a in alleles for pos,p in windows]
    return dict(sequence=sequence, alleles=alleles, arm=arm, window_count=len(windows), rows=results)


def parse_batch(text):
    records = list(csv.reader(io.StringIO(text.lstrip('\ufeff'))))
    records = [r for r in records if any(c.strip() for c in r)]
    if records and [c.strip().lower() for c in records[0]]==['peptide','allele']:
        records = records[1:]
    if not 1 <= len(records) <= MAX_BATCH:
        raise ValueError(f'Provide 1–{MAX_BATCH} pairs, with columns peptide,allele.')
    pairs = []
    for i,row in enumerate(records, 1):
        if len(row)!=2:
            raise ValueError(f'Row {i}: expected exactly two columns: peptide,allele.')
        try:
            pairs.append(validate_pair(*row))
        except ValueError as exc:
            raise ValueError(f'Row {i}: {exc}') from exc
    return pairs


def bootstrap():
    state = dataset()
    counts = state['counts']
    df = state['df']
    examples = []
    for a in [max(counts,key=counts.get), min(counts,key=counts.get)]:
        examples.append(dict(peptide=df[df.allele==a].peptide.iloc[0], allele=a, measurements=counts[a]))
    return dict(alleles=[dict(name=a, measurements=int(counts[a])) for a in sorted(counts)],
                models=[dict(id=a,name=n,scan=a in BASELINES) for a,n in MODEL_NAMES.items()],
                examples=examples, example_sequence=EXAMPLE_SEQUENCE,
                limits=dict(sequence=MAX_SEQUENCE, alleles=MAX_ALLELES, batch=MAX_BATCH),
                dataset=dict(rows=len(df),alleles=len(counts),zeros=int(df.thalf_hours.eq(0).sum())))
