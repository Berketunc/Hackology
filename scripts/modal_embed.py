"""Modal app: one-time frozen ESM-2 embedding extraction on GPU.

Two entrypoints:
  modal run scripts/modal_embed.py::benchmark
      Embeds a small sample, reports GPU memory + throughput + cost
      estimate. Run this FIRST to sanity-check before the full job.
  modal run scripts/modal_embed.py::full
      Embeds all eligible peptides + the 10 MHC heavy chains once and
      writes data/processed/embeddings_esm2.npz (+ _meta.json).

Checkpoint: facebook/esm2_t33_650M_UR50D (general pretrained protein
model; frozen; no stability-label supervision - see
reports/overlap_report.json).

Pooling: mean over residue tokens (BOS/EOS excluded), last hidden
layer. Pair feature downstream = concat(peptide_emb, mhc_emb).
"""

import json
import time
from pathlib import Path

import modal

WS = Path("/Users/berketunc/Hackology")
CHECKPOINT = "facebook/esm2_t33_650M_UR50D"
GPU = "T4"

hf_token = ""
env_file = Path.home() / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy")
)
# Always create a Secret object (even empty) so the function's
# dependency list is identical locally and remotely.
secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_token or "unset"})]

app = modal.App("hackology-esm2-embed", image=image)


def load_inputs():
    import pandas as pd
    elig = pd.read_csv(WS / "data" / "processed" / "eligible_records.csv",
                       dtype=str).fillna("")
    peptides = sorted(elig["peptide"].unique())
    train = pd.read_csv(
        WS / "data" / "external" / "spearmint_uq_train.csv", dtype=str)
    mhc = {}
    for allele in sorted(elig["allele"].unique()):
        seqs = train.loc[train["allele"] == allele,
                         "mhc_sequence"].unique()
        if len(seqs) != 1:
            raise ValueError(f"{allele}: {len(seqs)} sequences")
        mhc[allele] = seqs[0]
    return peptides, mhc


def _embed_impl(sequences, batch_size, use_gpu):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if use_gpu else "cpu"
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModel.from_pretrained(CHECKPOINT).to(device).eval()

    out = np.empty((len(sequences), model.config.hidden_size),
                   dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0
            for j, seq in enumerate(batch):
                mask[j, len(seq) + 1] = 0
            m = mask.unsqueeze(-1).float()
            out[i:i + len(batch)] = (
                (hidden * m).sum(1) / m.sum(1)).cpu().numpy()
    elapsed = time.time() - t0
    return {
        "embeddings": out,
        "elapsed_s": elapsed,
        "seqs_per_s": len(sequences) / elapsed,
        "gpu_mem_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if use_gpu else None),
        "gpu_name": (
            torch.cuda.get_device_name(0) if use_gpu else "cpu"),
        "hidden_size": model.config.hidden_size,
    }


@app.function(gpu=GPU, secrets=secrets, timeout=3600)
def embed(sequences, batch_size=64):
    return _embed_impl(sequences, batch_size, use_gpu=True)


@app.function(cpu=8.0, memory=16384, secrets=secrets, timeout=3600)
def embed_cpu(sequences, batch_size=32):
    return _embed_impl(sequences, batch_size, use_gpu=False)


@app.local_entrypoint()
def benchmark(device="cpu"):
    peptides, mhc = load_inputs()
    sample = peptides[:512] + list(mhc.values())
    fn = embed if device == "gpu" else embed_cpu
    stats = fn.remote(sample)
    n_total = len(peptides) + len(mhc)
    est_s = n_total / stats["seqs_per_s"]
    # Approx. list prices: T4 ~$0.59/hr; 8 vCPU ~ $0.27/hr.
    rate = 0.59 if device == "gpu" else 0.27
    print(json.dumps({
        "checkpoint": CHECKPOINT, "device": device,
        "gpu_name": stats["gpu_name"],
        "sample_seqs": len(sample),
        "elapsed_s": round(stats["elapsed_s"], 2),
        "seqs_per_s": round(stats["seqs_per_s"], 1),
        "peak_gpu_mem_gb": stats["gpu_mem_gb"],
        "n_total_seqs": n_total,
        "est_full_seconds": round(est_s, 1),
        "est_full_cost_usd": round(est_s / 3600 * rate, 4),
    }, indent=2))


@app.local_entrypoint()
def full(device="cpu"):
    peptides, mhc = load_inputs()
    fn = embed if device == "gpu" else embed_cpu
    # Fan out peptides across workers; each container loads the model
    # once (~2.6 GB fp32) then embeds its chunk.
    n_workers = 1 if device == "gpu" else 8
    chunks = [
        peptides[i::n_workers] for i in range(n_workers) if peptides[i::n_workers]
    ]
    results = list(fn.map(chunks))
    import numpy as np
    pep_emb = np.empty((len(peptides), results[0]["embeddings"].shape[1]),
                       dtype=np.float32)
    elapsed = 0.0
    for i, res in enumerate(results):
        idx = list(range(i, len(peptides), n_workers))
        pep_emb[idx] = res["embeddings"]
        elapsed += res["elapsed_s"]
    alleles = sorted(mhc)
    mhc_res = fn.remote([mhc[a] for a in alleles])
    out = WS / "data" / "processed"
    np.savez_compressed(
        out / "embeddings_esm2.npz",
        peptides=np.array(peptides), peptide_emb=pep_emb,
        alleles=np.array(alleles), mhc_emb=mhc_res["embeddings"],
    )
    meta = {
        "checkpoint": CHECKPOINT,
        "pooling": "mean over residue tokens (BOS/EOS excluded), "
                   "last hidden layer",
        "frozen": True,
        "infra": f"Modal {device} ({results[0]['gpu_name']}, "
                 f"{len(chunks)} workers)",
        "peptides": len(peptides),
        "alleles": {a: {"length": len(mhc[a])} for a in alleles},
        "mhc_sequence_source": "spearmint_uq_train.csv",
        "a0201_variant": "canonical (variant 0); engineered K90A/E87Q "
                         "variants excluded",
        "peptide_embed_worker_seconds_total": elapsed,
        "peak_gpu_mem_gb": results[0]["gpu_mem_gb"],
    }
    (out / "embeddings_meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", out / "embeddings_esm2.npz")
    print(json.dumps(meta, indent=2))
