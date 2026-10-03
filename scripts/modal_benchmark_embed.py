"""Frozen general ESM-2 arms for the organiser data; no label input to workers.
Run: python -m scripts.modal_benchmark_embed
Synthetic linker is GGGG. Positions 1:10 exclude BOS and capture the 9 peptide residues.
"""
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal
import numpy as np
from src.config import CHECKPOINT, HIDDEN_SIZE, DATA, RESULTS

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.1','transformers==5.18.0','numpy==2.4.6')
         .env({'HF_HOME':'/model-cache'})
         .add_local_python_source('src'))
volume = modal.Volume.from_name('hackology-general-esm2-cache', create_if_missing=True)
app = modal.App('hackology-rasmussen-representations', image=image)
_MODEL = None
_TOKENIZER = None


@app.function(gpu='L4', volumes={'/model-cache':volume}, timeout=3600, max_containers=1, memory=16384)
def extract(sequences, mode):
    import torch
    from transformers import AutoTokenizer, AutoModel
    global _MODEL, _TOKENIZER
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
        _MODEL = AutoModel.from_pretrained(CHECKPOINT, dtype=torch.float16).to('cuda').eval()
    loaded = time.perf_counter()
    assert _MODEL.config.hidden_size == HIDDEN_SIZE
    outputs = []
    with torch.inference_mode():
        for pos in range(0, len(sequences), 64):
            batch = sequences[pos:pos+64]
            tokens = _TOKENIZER(batch, padding=True, return_tensors='pt').to('cuda')
            hidden = _MODEL(**tokens).last_hidden_state
            if mode == 'joint':
                value = hidden[:, 1:10].reshape(len(batch), -1)
            else:
                mask = tokens.attention_mask.clone()
                mask[:, 0] = 0
                for j, seq in enumerate(batch):
                    mask[j,len(seq)+1] = 0
                value = (hidden.float()*mask.unsqueeze(-1)).sum(1) / mask.sum(1,keepdim=True)
            outputs.append(value.float().cpu().numpy())
    torch.cuda.synchronize()
    end = time.perf_counter()
    return dict(features=np.concatenate(outputs), gpu_seconds=end-start,
                inference_seconds=end-loaded, model_load_seconds=loaded-start,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                device=torch.cuda.get_device_name(), sequences=len(sequences))


def main():
    from src.splits import prepare
    from src.run_benchmark import lock_design
    print('Preparing locked dataset',flush=True)
    df = prepare()
    lock_design()
    manifest = json.loads((DATA/'manifests/rasmussen_manifest.json').read_text())
    for arm, mode in [('esm2_mean','mean'),('esm2_joint','joint')]:
        target = DATA/'processed'/f'{arm}.npy'
        from src.features import embedding_arm_available
        if embedding_arm_available(arm):
            from src.features import load_embedding_arm
            load_embedding_arm(arm, df)
            print(f'Validated existing {arm}',flush=True)
            continue
        if mode == 'mean':
            peptides = sorted(df.peptide.unique())
            hla = sorted(df.hla_pseudoseq.unique())
            seqs = peptides + hla
        else:
            seqs = (df.peptide + 'GGGG' + df.hla_pseudoseq).tolist()
        start = time.perf_counter()
        stats, values = [], []
        print(f'Starting Modal extraction: {arm}, {len(seqs)} sequences',flush=True)
        with modal.enable_output(), app.run():
            for offset in range(0,len(seqs),2048):
                res = extract.remote(seqs[offset:offset+2048], mode)
                values.append(res.pop('features'))
                stats.append(res)
                print(f'{arm}: {min(offset+2048,len(seqs))}/{len(seqs)}; {res}',flush=True)
        arr = np.concatenate(values)
        if mode == 'mean':
            pepmap = {p: arr[i] for i,p in enumerate(peptides)}
            hlamap = {h: arr[len(peptides)+i] for i,h in enumerate(hla)}
            arr = np.stack([np.concatenate([pepmap[p],hlamap[h]]) for p,h in zip(df.peptide,df.hla_pseudoseq)])
        assert np.isfinite(arr).all()
        np.save(target,arr)
        np.save(target.with_name(f'{arm}_row_ids.npy'),df.index.to_numpy())
        meta = dict(checkpoint=CHECKPOINT, hidden_size=HIDDEN_SIZE, feature_dimension=arr.shape[1],
                    rows=len(arr), frozen=True, dtype='float16 inference; float32 output',
                    pooling='residue mean' if mode=='mean' else 'concatenated 9 peptide hidden states',
                    hla_input='34-residue pseudosequence', linker='GGGG' if mode=='joint' else None,
                    dataset_sha256=manifest['sha256'], gpu_seconds=sum(s['gpu_seconds'] for s in stats),
                    inference_seconds=sum(s['inference_seconds'] for s in stats),
                    peak_gpu_memory_bytes=max(s['peak_gpu_memory_bytes'] for s in stats),
                    local_wall_seconds=time.perf_counter()-start, hardware=stats[0]['device'], batches=stats,
                    cost_scope='GPU function wall time incl model load; excludes container startup and image build; not a billing total')
        meta['representation_name']='independent mean pooling' if mode=='mean' else 'joint peptide–HLA sequence encoding'
        meta['extracted_sequences']=len(seqs)
        meta['inference_sequences_per_second']=len(seqs)/meta['inference_seconds']
        meta['gpu_function_sequences_per_second']=len(seqs)/meta['gpu_seconds']
        target.with_name(f'{arm}_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f'Saved {arm}: {arr.shape}',flush=True)


if __name__ == '__main__':
    main()
