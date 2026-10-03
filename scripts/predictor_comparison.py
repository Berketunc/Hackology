"""Predictor comparison under matched random acquisition.

One-hot + Ridge vs frozen ESM-2 + Ridge on identical partitions,
identical random label orders, identical budgets, identical fitting
procedure (Ridge alpha=10, solver=lsqr). The ONLY difference is the
input representation.

Predefined budget grid (pair-level labels):
    [173, 346, 519, 692, 865, 1038, 1211, 1384, 1557, 1728, 1903, 2076]
= 5%..60% of the 3,457-pair acquisition pool.

Primary metric (predefined): development-set RMSE in log1p(hours).
Final test partition is not touched.

Unsupervised use of pool inputs (documented): DictVectorizer and
StandardScaler are fit on acquisition-pool INPUTS only; no pool labels
are used in preprocessing.
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, DATA, RESULTS, REPORTS
from src.features import validate_legacy_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

WS = ROOT
PARTITIONS = WS / "results" / "pilot_all_alleles" / "partitioned_data.csv"
EMB = WS / "data" / "processed" / "embeddings_esm2.npz"
OUT = WS / "results" / "predictor_comparison"

BUDGETS = [173, 346, 519, 692, 865, 1038, 1211, 1384, 1557, 1728,
           1903, 2076]
SEEDS = range(10)
ALPHA = 10.0


def onehot_dicts(data):
    result = []
    for row in data.itertuples():
        item = {
            f"allele={row.allele}": 1.0,
            f"length={len(row.peptide)}": 1.0,
        }
        for position, aa in enumerate(row.peptide):
            item[f"position={position}:aa={aa}"] = 1.0
        result.append(item)
    return result


def main():
    data = pd.read_csv(PARTITIONS, dtype=str).fillna("")
    data["hours"] = pd.to_numeric(data["half_life_hours"])

    pool = data["partition"].eq("acquisition_pool").to_numpy()
    dev = data["partition"].eq("development").to_numpy()
    y_pool = np.log1p(data.loc[pool, "hours"].to_numpy())
    y_dev = np.log1p(data.loc[dev, "hours"].to_numpy())

    # --- one-hot representation (as in the pilot) ---
    vec = DictVectorizer(sparse=True)
    vec.fit(onehot_dicts(data.loc[pool]))
    x_all_oh = vec.transform(onehot_dicts(data))
    x_pool_oh, x_dev_oh = x_all_oh[pool], x_all_oh[dev]

    # --- ESM-2 pair representation: concat(peptide, mhc) ---
    validate_legacy_cache()
    emb = np.load(EMB, allow_pickle=False)
    pep_lookup = dict(zip(emb["peptides"].tolist(), emb["peptide_emb"]))
    mhc_lookup = dict(zip(emb["alleles"].tolist(), emb["mhc_emb"]))
    x_all_esm = np.vstack([
        np.concatenate([pep_lookup[p], mhc_lookup[a]])
        for p, a in zip(data["peptide"], data["allele"])
    ]).astype(np.float32)
    scaler = StandardScaler().fit(x_all_esm[pool])
    x_pool_esm = scaler.transform(x_all_esm[pool])
    x_dev_esm = scaler.transform(x_all_esm[dev])

    runs = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(pool.sum())
        for budget in BUDGETS:
            chosen = order[:budget]
            for name, xp, xd in [
                ("onehot_ridge", x_pool_oh, x_dev_oh),
                ("esm2_ridge", x_pool_esm, x_dev_esm),
            ]:
                model = Ridge(alpha=ALPHA, solver="lsqr")
                model.fit(xp[chosen], y_pool[chosen])
                pred = model.predict(xd)
                runs.append({
                    "seed": seed, "labels_acquired": budget,
                    "predictor": name,
                    "rmse_log1p": float(np.sqrt(
                        mean_squared_error(y_dev, pred))),
                    "mae_log1p": float(np.mean(np.abs(y_dev - pred))),
                })

    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / "runs.csv", index=False)
    summary = (
        runs.groupby(["predictor", "labels_acquired"])
        .agg(rmse_mean=("rmse_log1p", "mean"),
             rmse_sd=("rmse_log1p", "std"),
             mae_mean=("mae_log1p", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.pivot(index="labels_acquired", columns="predictor",
                        values="rmse_mean").to_string())

    meta = {
        "design": "matched random acquisition; identical Ridge(alpha=10,"
                  "lsqr) for both representations",
        "budgets": BUDGETS, "seeds": list(SEEDS),
        "primary_metric": "development RMSE in log1p(hours)",
        "partitions": str(PARTITIONS),
        "final_test_evaluated": False,
    }
    (OUT / "design.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
