"""Three-policy active-learning comparison.

Predefined design (locked before running):
- Predictor: RidgeCV(alphas=[1,10,100,300,1000,3000], LOO-CV) on
  StandardScaler-transformed concat(peptide_ESM2, mhc_ESM2) features.
  Regularisation is selected on acquired labels only, each round.
  IDENTICAL for every policy - only the acquisition order differs.
  (Fixed alpha=10 was shown to be mis-scaled for the 2560-dim
  embedding space in the predictor-comparison diagnostic.)
- Initial labelled set: first 173 pool pairs of the seed's random
  permutation - identical across all three policies within a seed.
- Batch size: 173 pairs per round; budgets = 173*k, k = 1..12
  (5%..60% of the 3,457-pair pool).
- Policies:
    random      - seeded permutation.
    seqdiv      - greedy k-center (max-min Euclidean) in the one-hot
                  sequence space (allele + length + position:aa).
    embdiv      - greedy k-center (max-min Euclidean) in the
                  standardized ESM-2 pair space.
  Diversity scores are recomputed each round against the current
  labelled set, and within a batch against points already selected.
- Seeds: 0..9 (10).
- Primary metric: development-set RMSE in log1p(hours) at each budget.
  Scalar summary: area under RMSE vs log10(labels) curve per
  (policy, seed); paired differences vs random.
- Final test partition is NOT touched.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

WS = Path("/Users/berketunc/Hackology")
PARTITIONS = WS / "results" / "pilot_all_alleles" / "partitioned_data.csv"
EMB = WS / "data" / "processed" / "embeddings_esm2.npz"
OUT = WS / "results" / "policy_comparison"

INIT = 173
BATCH = 173
N_ROUNDS = 11          # budgets: 173 + 11*173 = 2076 (~60% of pool)
SEEDS = range(10)
ALPHAS = [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]


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


def greedy_kcenter(candidates, labelled, feat, batch):
    """Max-min greedy pick of `batch` indices from `candidates`.

    Maintains a running min-distance vector from candidates to
    (labelled + already-picked), updated incrementally per pick.
    """
    cand = np.asarray(candidates)
    dmin = cdist(feat[cand], feat[labelled]).min(axis=1)
    picked = []
    for _ in range(batch):
        i = int(np.argmax(dmin))
        picked.append(int(cand[i]))
        # distance to the newly picked point
        d_new = cdist(feat[cand], feat[[cand[i]]]).ravel()
        keep = np.ones(len(cand), dtype=bool)
        keep[i] = False
        cand = cand[keep]
        dmin = np.minimum(dmin[keep], d_new[keep])
    return picked


def main():
    data = pd.read_csv(PARTITIONS, dtype=str).fillna("")
    data["hours"] = pd.to_numeric(data["half_life_hours"])
    pool = np.where(data["partition"].eq("acquisition_pool"))[0]
    dev = np.where(data["partition"].eq("development"))[0]
    y_pool = np.log1p(data["hours"].to_numpy()[pool])
    y_dev = np.log1p(data["hours"].to_numpy()[dev])

    # Predictor features: ESM-2 pair space (all policies).
    emb = np.load(EMB, allow_pickle=False)
    pep_lookup = dict(zip(emb["peptides"].tolist(), emb["peptide_emb"]))
    mhc_lookup = dict(zip(emb["alleles"].tolist(), emb["mhc_emb"]))
    x_all = np.vstack([
        np.concatenate([pep_lookup[p], mhc_lookup[a]])
        for p, a in zip(data["peptide"], data["allele"])
    ]).astype(np.float64)
    scaler = StandardScaler().fit(x_all[pool])
    x_esm = scaler.transform(x_all)
    x_pool, x_dev = x_esm[pool], x_esm[dev]

    # Acquisition spaces.
    vec = DictVectorizer(sparse=False)
    seq_space = vec.fit_transform(onehot_dicts(data))[pool]
    emb_space = x_pool  # standardized ESM-2 pair features

    rng_time = time.time()
    runs = []
    selections = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pool))
        init = list(order[:INIT])

        # Per-policy acquisition state (indices into pool arrays).
        labelled = {
            "random": list(init),
            "seqdiv": list(init),
            "embdiv": list(init),
        }
        remaining = {
            name: list(order[INIT:]) for name in labelled
        }

        for step in range(N_ROUNDS + 1):
            budget = INIT + step * BATCH
            for name in labelled:
                chosen = np.array(labelled[name])
                model = RidgeCV(alphas=ALPHAS)
                model.fit(x_pool[chosen], y_pool[chosen])
                pred = model.predict(x_dev)
                runs.append({
                    "seed": seed, "policy": name,
                    "labels_acquired": budget,
                    "rmse_log1p": float(np.sqrt(
                        mean_squared_error(y_dev, pred))),
                    "mae_log1p": float(np.mean(np.abs(y_dev - pred))),
                })
            if step == N_ROUNDS:
                break
            # acquire next batch per policy
            labelled["random"] += remaining["random"][:BATCH]
            remaining["random"] = remaining["random"][BATCH:]
            for name, space in [("seqdiv", seq_space),
                                ("embdiv", emb_space)]:
                picked = greedy_kcenter(
                    remaining[name], labelled[name], space, BATCH)
                labelled[name] += picked
                selections += [
                    {"seed": seed, "policy": name, "step": step + 1,
                     "pool_index": int(p)} for p in picked
                ]
                sel = set(picked)
                remaining[name] = [c for c in remaining[name]
                                   if c not in sel]
        print(f"seed {seed} done ({time.time() - rng_time:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)

    summary = (
        runs.groupby(["policy", "labels_acquired"])
        .agg(rmse_mean=("rmse_log1p", "mean"),
             rmse_sd=("rmse_log1p", "std"),
             mae_mean=("mae_log1p", "mean"))
        .reset_index()
    )
    summary.to_csv(OUT / "summary.csv", index=False)

    # Scalar: area under RMSE-vs-log10(labels) per (policy, seed).
    aulc = []
    for (pol, seed), g in runs.groupby(["policy", "seed"]):
        g = g.sort_values("labels_acquired")
        x = np.log10(g["labels_acquired"].to_numpy())
        area = np.trapezoid(g["rmse_log1p"].to_numpy(), x)
        aulc.append({"policy": pol, "seed": seed, "aulc": float(area)})
    aulc = pd.DataFrame(aulc)
    aulc.to_csv(OUT / "aulc.csv", index=False)

    paired = aulc.pivot(index="seed", columns="policy",
                        values="aulc")
    print("\nDev RMSE by policy/budget:")
    print(summary.pivot(index="labels_acquired", columns="policy",
                        values="rmse_mean").to_string())
    print("\nAULC (lower=better), per policy:")
    print(aulc.groupby("policy")["aulc"].agg(["mean", "std"]).to_string())
    print("\nPaired AULC differences vs random (negative=better):")
    for pol in ["seqdiv", "embdiv"]:
        diff = paired[pol] - paired["random"]
        print(f"  {pol}: mean={diff.mean():.4f} sd={diff.std():.4f} "
              f"seeds_better={int((diff < 0).sum())}/{len(diff)}")

    meta = {
        "init": INIT, "batch": BATCH, "n_rounds": N_ROUNDS,
        "seeds": list(SEEDS), "alphas": ALPHAS,
        "predictor": "RidgeCV(LOO) on standardized concat(peptide,mhc)"
                     " ESM-2 t33_650M features",
        "policies": ["random", "seqdiv", "embdiv"],
        "primary_metric": "development RMSE in log1p(hours)",
        "scalar_summary": "AULC over log10(labels)",
        "final_test_evaluated": False,
    }
    (OUT / "design.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
