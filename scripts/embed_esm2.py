"""Extract frozen ESM-2 embeddings for the eligible subset.

Model: facebook/esm2_t30_150M_UR50D (general pretrained protein model;
NOT stability-trained - see reports/overlap_report.json for why a
SPEARMINT stability checkpoint is disallowed).

Representation: mean-pooled last hidden layer over residue tokens
(BOS/EOS excluded), one embedding per unique peptide and per unique
full-length MHC heavy-chain sequence. Pair feature used downstream is
concat(peptide_emb, mhc_emb).

MHC sequences come from the SPEARMINT split files (full-length heavy
chains, 362-365 aa). For HLA-A*02:01 the canonical variant is used;
K90A and E87Q/K90A engineered variants appear only in the s3 test file
and are not part of this dataset.

Output: data/processed/embeddings_esm2.npz + embeddings_meta.json
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

WS = Path("/Users/berketunc/Hackology")
CHECKPOINT = "facebook/esm2_t30_150M_UR50D"

# Canonical full-length heavy chains from SPEARMINT split files
# (variant 0 for A*02:01 - see module docstring).
MHC_SEQ_FILE = WS / "data" / "external" / "spearmint_uq_train.csv"


def main():
    elig = pd.read_csv(WS / "data" / "processed" / "eligible_records.csv",
                       dtype=str).fillna("")
    peptides = sorted(elig["peptide"].unique())

    mhc_map = {}
    train = pd.read_csv(MHC_SEQ_FILE, dtype=str)
    for allele in elig["allele"].unique():
        seqs = train.loc[train["allele"] == allele, "mhc_sequence"].unique()
        if len(seqs) != 1:
            raise ValueError(f"{allele}: {len(seqs)} sequences in train file")
        mhc_map[allele] = seqs[0]

    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"device={device}, checkpoint={CHECKPOINT}")

    def embed(sequences, batch_size=32):
        out = np.empty((len(sequences), model.config.hidden_size),
                       dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=False).to(device)
                hidden = model(**enc).last_hidden_state
                mask = enc["attention_mask"].clone()
                mask[:, 0] = 0                      # BOS
                for j, seq in enumerate(batch):     # EOS at len+1
                    mask[j, len(seq) + 1] = 0
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1)
                out[i:i + len(batch)] = pooled.cpu().numpy()
        return out

    t0 = time.time()
    pep_emb = embed(peptides)
    alleles = sorted(mhc_map)
    mhc_emb = embed([mhc_map[a] for a in alleles])
    print(f"embedded {len(peptides)} peptides + {len(alleles)} alleles "
          f"in {time.time() - t0:.0f}s")

    np.savez_compressed(
        WS / "data" / "processed" / "embeddings_esm2.npz",
        peptides=np.array(peptides), peptide_emb=pep_emb,
        alleles=np.array(alleles), mhc_emb=mhc_emb,
    )
    meta = {
        "checkpoint": CHECKPOINT,
        "pooling": "mean over residue tokens (BOS/EOS excluded), "
                   "last hidden layer",
        "frozen": True,
        "device": device,
        "peptides": len(peptides),
        "alleles": {a: {"length": len(mhc_map[a])} for a in alleles},
        "mhc_sequence_source": str(MHC_SEQ_FILE.name),
        "a0201_variant": "canonical (variant 0); engineered K90A/E87Q "
                         "variants excluded",
    }
    (WS / "data" / "processed" / "embeddings_meta.json").write_text(
        json.dumps(meta, indent=2))
    print("saved embeddings_esm2.npz + embeddings_meta.json")


if __name__ == "__main__":
    main()
