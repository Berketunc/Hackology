"""Honest inference cards using saved CV fits; no pretrained supervised checkpoint."""
from functools import lru_cache
import json
import joblib
import numpy as np
import pandas as pd
import torch
from .config import DATA, RESULTS, CHECKPOINT, HIDDEN_SIZE
from .data import main_data, AA
from .features import blosum_encode, load_embedding_arm, onehot_allele


def resources():
    df = main_data()
    rows = pd.read_csv(RESULTS/'benchmark/per_row.csv')
    return df, rows


@lru_cache(maxsize=30)
def model(arm, fold):
    return joblib.load(RESULTS/'models'/f'{arm}_peptide_{fold}.joblib')


@lru_cache(maxsize=2)
def embeddings(arm):
    return load_embedding_arm(arm,resources()[0])


@lru_cache(maxsize=1)
def local_plm():
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(CHECKPOINT)
    net=AutoModel.from_pretrained(CHECKPOINT).eval()
    assert net.config.hidden_size == HIDDEN_SIZE
    return tok,net


@lru_cache(maxsize=32)
def new_pair_features(peptide,pseudoseq):
    tok,net=local_plm()
    with torch.inference_mode():
        mean=[]
        for seq in [peptide,pseudoseq]:
            hidden=net(**tok(seq,return_tensors='pt')).last_hidden_state
            mean.append(hidden[0,1:len(seq)+1].mean(0).numpy())
        hidden=net(**tok(peptide+'GGGG'+pseudoseq,return_tensors='pt')).last_hidden_state
    return {'esm2_mean':np.concatenate(mean)[None,:],
            'esm2_joint':hidden[:,1:10].reshape(1,-1).numpy()}


def overlap_lookup(peptide,allele):
    path=DATA/'processed/rasmussen_overlap.csv'
    if not path.exists():
        return 'SPEARMINT exposure audit unavailable; no claim of absence. General ESM-2 sequence exposure is unknown.'
    frame=pd.read_csv(path)
    match=frame[(frame.peptide==peptide)&(frame.allele==allele)]
    if len(match):
        row=match.iloc[0]
        names=[key.removeprefix('in_') for key in frame.columns if key.startswith('in_') and bool(row[key])]
        training=[n for n in names if n in ['spearmint_uq_train','spearmint_uq_s3_train']]
        status=('Yes: '+', '.join(training)) if training else 'No match in the two audited SPEARMINT training files; other training exposure is not ruled out'
        return 'Published SPEARMINT training-file match: '+status+'. All matching split files: '+(', '.join(names) or 'none')+'. ESM-2 pretraining sequence exposure is unknown. No SPEARMINT weights are used.'
    return 'Pair is absent from this organiser dataset; external model-training exposure is unknown.'


def predict(peptide,allele):
    peptide=peptide.strip().upper()
    if len(peptide)!=9 or not set(peptide)<=AA:
        raise ValueError('Enter exactly 9 standard amino-acid letters (ACDEFGHIKLMNPQRSTVWY).')
    df,rows=resources()
    if allele not in set(df.allele):
        raise ValueError('Select an allele from the dataset.')
    allele_rows=df[df.allele==allele]
    pseudo=allele_rows.hla_pseudoseq.iloc[0]
    pair=df[(df.peptide==peptide)&(df.allele==allele)]
    counts=df.groupby('allele').size()
    eligible=counts[counts>counts[allele]].index
    seqs=df.drop_duplicates('allele').set_index('allele').hla_pseudoseq
    nearest='None: this allele has the largest measurement count.'
    if len(eligible):
        ids=seqs.loc[eligible].map(lambda s:sum(a==b for a,b in zip(s,pseudo))/34)
        best=ids.idxmax()
        nearest=f'{best} ({int(counts[best])} dataset measurements; {ids[best]:.1%} contact-residue identity)'
    arms=[a for a in ['blosum_nn','blosum_ridge','onehot_ridge','esm2_mean','esm2_joint'] if all((RESULTS/'models'/f'{a}_peptide_{f}.joblib').exists() for f in range(5)) and len(rows[(rows.arm==a)&(rows.split_regime=='peptide')])==len(df)]
    known=not pair.empty
    new_features=None
    output=[]
    for arm in arms:
        if arm.startswith('esm2'):
            if known:
                pos=df.index.get_loc(pair.index[0])
                x=np.asarray(embeddings(arm)[pos:pos+1])
            else:
                if new_features is None:
                    new_features=new_pair_features(peptide,pseudo)
                x=new_features[arm]
        elif arm.startswith('blosum'):
            x=np.concatenate([blosum_encode(peptide),blosum_encode(pseudo)])[None,:]
        predictions=[]
        for fold in range(5):
            bundle=model(arm,fold)
            if arm=='onehot_ridge':
                encoded,_=onehot_allele(pd.DataFrame({'allele':[allele]}),bundle['encoder'])
                x=np.column_stack([blosum_encode(peptide)[None,:],encoded])
            predictions.append(float(bundle['model'].predict(x)[0]))
        if known:
            oof=rows[(rows.row_index==pair.index[0])&(rows.arm==arm)&(rows.split_regime=='peptide')]
            center=float(oof.y_pred.iloc[0])
            central_fold=int(oof.fold.iloc[0])
            training_ids=model(arm,central_fold)['row_indices']
            allele_train_count=int(df.loc[training_ids].allele.eq(allele).sum())
        else:
            center=float(np.mean(predictions))
            allele_train_count=int(round(np.mean([df.loc[model(arm,f)['row_indices']].allele.eq(allele).sum() for f in range(5)])))
        hours=float(np.expm1(max(0,center)))
        low,high=np.expm1(np.maximum(0,[min(predictions),max(predictions)]))
        tier='≥6 h' if hours>=6 else '2–6 h' if hours>=2 else '<2 h'
        labels={'blosum_nn':'BLOSUM MLP (reference)','blosum_ridge':'BLOSUM Ridge','onehot_ridge':'Allele-ID Ridge (floor)','esm2_mean':'ESM-2 independent mean','esm2_joint':'ESM-2 joint sequence encoding'}
        output.append([labels[arm],round(hours,2),f'{low:.2f}–{high:.2f}',tier,allele_train_count])
    info=(f'### {allele} · {peptide}\n\n**{int(counts[allele])} recorded measurements in this dataset** for this allele. '
          f'Nearest better-measured neighbour: {nearest}\n\n'
          +('Central predictions come from the fold that held this peptide out. ' if known else 'Novel-pair predictions average five fitted models. ')
          +'The range is the minimum–maximum across five peptide-CV fits, **not a calibrated confidence interval**. '
          +('Four fits can include this measured pair; their spread is only a model-sensitivity diagnostic. ' if known else '')
          +'Training counts refer to available outer-fold rows; the neural model reserves 15% of training groups for early stopping. Negative log-scale outputs are clipped to zero hours for display only.\n\n'
          +'**Training exposure:** '+overlap_lookup(peptide,allele))
    if known:
        info+=f'\n\nRecorded half-life: {pair.thalf_hours.iloc[0]:g} h (shown for retrospective comparison).'
    zero_fraction=float(allele_rows.thalf_hours.eq(0).mean())
    info+=f'\n\nReported zeros for this allele: {zero_fraction:.1%}. '
    if zero_fraction>=.5:
        info+='At least half of its measurements are zero; consult tier-AUC alongside ranking correlation in the benchmark report. '
    info+='Dataset measurement counts do not describe human population representation.'
    return info,pd.DataFrame(output,columns=['Arm','Predicted half-life (h)','Five-fit range (h)','Tier','Allele rows in training fold'])


def build_overlap():
    from archive.iedb_pilot.scripts.overlap_report import pairs
    df=main_data()
    out=df[['allele','peptide']].copy()
    for name in ['uq_train','uq_bs_val','uq_bs_test','uq_s3_train','uq_s3_val','uq_s3_test']:
        path=DATA/'external'/f'spearmint_{name}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Cannot audit exposure without {path}')
        source=pd.read_csv(path,usecols=['allele','peptide_sequence'])
        keys=pairs(source,'allele','peptide_sequence')
        out[f'in_spearmint_{name}']=[key in keys for key in zip(out.allele,out.peptide)]
    out.to_csv(DATA/'processed/rasmussen_overlap.csv',index=False)
    return out
