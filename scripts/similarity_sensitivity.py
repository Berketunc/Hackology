"""Sequence-similarity sensitivity analysis.

Question: are the dev-set results inflated by near-sequence overlap
between acquisition pool and dev peptides? Exact-peptide grouping was
enforced at partition time; homologous peptides were not.

Identity definition (documented): ungapped identity of the SHORTER
peptide aligned at its best-matching offset within the longer;
equal-length peptides use matches/length (i.e. 1 - Hamming/L).

Outputs:
  results/sensitivity/dev_max_identity_to_pool.csv
      per dev peptide, max identity to any pool peptide
  results/sensitivity/stratified_rmse.csv
      per (policy, seed, threshold): dev RMSE restricted to dev
      peptides whose max identity to the ACQUIRED set is < threshold,
      at the final budget (2,076 labels)
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, DATA, RESULTS, REPORTS
from src.features import validate_legacy_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

WS = ROOT
PARTITIONS = WS / "results" / "pilot_all_alleles" / "partitioned_data.csv"
EMB = WS / "data" / "processed" / "embeddings_esm2.npz"
SEL = WS / "results" / "policy_comparison" / "selections.csv"
OUT = WS / "results" / "sensitivity"

INIT = 173
FINAL_BUDGET = 2076
SEEDS = range(10)
ALPHAS = [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]
THRESHOLDS = [1.0, 0.9, 0.8, 0.7]


def max_identity_matrix(short_arr, long_arr):
    """(Ns, Nl) best ungapped identity; rows shorter-or-equal.

    If len(short) == len(long): Hamming identity.
    Else: slide `short` along `long`, max matches/len(short).
    """
    ls, ll = short_arr.shape[1], long_arr.shape[1]
    if ls == ll:
        return (short_arr[:, None, :] == long_arr[None, :, :]).mean(-1)
    if ls > ll:
        return max_identity_matrix(long_arr, short_arr).T
    best = np.zeros((len(short_arr), len(long_arr)))
    for off in range(ll - ls + 1):
        ident = (short_arr[:, None, :]
                 == long_arr[None, :, off:off + ls]).mean(-1)
        best = np.maximum(best, ident)
    return best


def pairwise_max_identity(seqs_a, seqs_b):
    """(Na, Nb) max ungapped identity, grouping by length pair."""
    seqs_a, seqs_b = list(seqs_a), list(seqs_b)
    la = np.array([len(s) for s in seqs_a])
    lb = np.array([len(s) for s in seqs_b])
    out = np.zeros((len(seqs_a), len(seqs_b)))
    for len_a in np.unique(la):
        ia = np.where(la == len_a)[0]
        a = np.array([list(seqs_a[i]) for i in ia], dtype="U1")
        for len_b in np.unique(lb):
            ib = np.where(lb == len_b)[0]
            b = np.array([list(seqs_b[j]) for j in ib], dtype="U1")
            out[np.ix_(ia, ib)] = max_identity_matrix(a, b)
    return out


def main():
    data = pd.read_csv(PARTITIONS, dtype=str).fillna("")
    data["hours"] = pd.to_numeric(data["half_life_hours"])
    pool_idx = np.where(data["partition"].eq("acquisition_pool"))[0]
    dev_idx = np.where(data["partition"].eq("development"))[0]
    pool_pep = data["peptide"].to_numpy()[pool_idx]
    dev_pep = data["peptide"].to_numpy()[dev_idx]
    y_pool = np.log1p(data["hours"].to_numpy()[pool_idx])
    y_dev = np.log1p(data["hours"].to_numpy()[dev_idx])

    validate_legacy_cache()
    emb = np.load(EMB, allow_pickle=False)
    pl = dict(zip(emb["peptides"].tolist(), emb["peptide_emb"]))
    ml = dict(zip(emb["alleles"].tolist(), emb["mhc_emb"]))
    x_all = np.vstack([
        np.concatenate([pl[p], ml[a]])
        for p, a in zip(data["peptide"], data["allele"])
    ])
    sc = StandardScaler().fit(x_all[pool_idx])
    x_pool, x_dev = sc.transform(x_all[pool_idx]), sc.transform(
        x_all[dev_idx])

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Intrinsic leakage ceiling: dev vs ALL pool peptides.
    ident_pool = pairwise_max_identity(dev_pep, pool_pep)
    dev_max_pool = ident_pool.max(axis=1)
    pd.DataFrame({
        "dev_peptide": dev_pep,
        "max_identity_to_pool": dev_max_pool,
        "dev_allele": data["allele"].to_numpy()[dev_idx],
    }).to_csv(OUT / "dev_max_identity_to_pool.csv", index=False)
    for t in [1.0, 0.9, 0.8, 0.7, 0.6]:
        frac = (dev_max_pool >= t).mean()
        print(f"dev peptides with a pool neighbour >= {t:.0%} "
              f"identity: {frac:.1%} ({int((dev_max_pool >= t).sum())}"
              f"/{len(dev_pep)})")

    # 2. Stratified dev RMSE at final budget, per policy/seed.
    sel = pd.read_csv(SEL)
    y_pool_arr = np.log1p(data["hours"].to_numpy())[pool_idx]
    rows = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(pool_idx))
        init = order[:INIT]
        labelled = {"random": list(order[:FINAL_BUDGET])}
        for pol in ["seqdiv", "embdiv"]:
            extra = sel.loc[
                (sel["seed"] == seed) & (sel["policy"] == pol),
                "pool_index"].tolist()
            labelled[pol] = list(init) + extra

        for pol, chosen_list in labelled.items():
            chosen = np.asarray(chosen_list[:FINAL_BUDGET])
            model = RidgeCV(alphas=ALPHAS).fit(
                x_pool[chosen], y_pool[chosen])
            err2 = (y_dev - model.predict(x_dev)) ** 2
            ident_acq = pairwise_max_identity(
                dev_pep, pool_pep[chosen]).max(axis=1)
            for t in THRESHOLDS:
                mask = ident_acq < t
                rows.append({
                    "seed": seed, "policy": pol, "threshold": t,
                    "n_dev": int(mask.sum()),
                    "rmse_log1p": float(np.sqrt(err2[mask].mean()))
                    if mask.any() else np.nan,
                })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "stratified_rmse.csv", index=False)
    print("\nDev RMSE at 2076 labels, restricted to peptides with "
          "max identity to acquired set < threshold:")
    print(res.pivot_table(index=["policy", "threshold"],
                          values=["rmse_log1p", "n_dev"],
                          aggfunc={"rmse_log1p": "mean",
                                   "n_dev": "mean"}).to_string())

    meta = {"identity": "best ungapped alignment of shorter peptide "
                        "into longer; matches/len(shorter)",
            "budget": FINAL_BUDGET, "seeds": list(SEEDS),
            "thresholds": THRESHOLDS}
    (OUT / "design.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
