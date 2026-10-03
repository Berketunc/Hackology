"""Overlap/provenance audit for the eligible subset.

Joins the 5,815 eligible (allele, peptide) pairs against:
  - the Rasmussen Google-Sheet snapshot (data/external/)
  - the SPEARMINT split CSVs promoted to data/external/

Purpose: quantify how much of our acquisition pool a SPEARMINT
stability-trained checkpoint has already seen, and confirm the IEDB
candidate subset's relationship to the Rasmussen dataset.
"""

import json
from pathlib import Path

import pandas as pd

EXT = Path("/Users/berketunc/Hackology/data/external")
PROC = Path("/Users/berketunc/Hackology/data/processed")
REPORTS = Path("/Users/berketunc/Hackology/reports")


def pairs(df, allele_col, pep_col):
    return set(zip(df[allele_col], df[pep_col]))


def main():
    elig = pd.read_csv(PROC / "eligible_records.csv", dtype=str).fillna("")
    elig_pairs = pairs(elig, "allele", "peptide")
    elig_hours = pd.to_numeric(elig["half_life_hours"])

    report = {"eligible_pairs": len(elig_pairs)}

    rasm = pd.read_csv(EXT / "rasmussen_et_al_dataset.csv")
    rasm_pairs = pairs(rasm, "allele", "peptide")
    inter = elig_pairs & rasm_pairs
    report["rasmussen"] = {
        "rows": len(rasm),
        "unique_pairs": len(rasm_pairs),
        "eligible_pairs_matched": len(inter),
        "eligible_pairs_unmatched": len(elig_pairs - rasm_pairs),
    }
    # Compare labels where both have numeric hours.
    rasm_key = rasm.set_index(["allele", "peptide"])["thalf_hours"]
    matched = elig.loc[
        elig.apply(lambda r: (r["allele"], r["peptide"]) in inter, axis=1)
    ].copy()
    matched["rasm_hours"] = matched.apply(
        lambda r: rasm_key.get((r["allele"], r["peptide"])), axis=1
    )
    both = matched.dropna(subset=["rasm_hours"]).copy()
    both["iedb_hours"] = pd.to_numeric(both["half_life_hours"])
    diff = (both["iedb_hours"] - both["rasm_hours"]).abs()
    report["rasmussen"]["label_agreement"] = {
        "compared": int(len(both)),
        "exact_match": int((diff == 0).sum()),
        "abs_diff_gt_0_01h": int((diff > 0.01).sum()),
        "max_abs_diff_hours": float(diff.max()) if len(diff) else None,
        "iedb_positive_rasmussen_zero": int(
            ((both["iedb_hours"] > 0) & (both["rasm_hours"] == 0)).sum()
        ),
    }

    spearmint = {}
    for name in ["uq_train", "uq_bs_val", "uq_bs_test",
                 "uq_s3_train", "uq_s3_val", "uq_s3_test"]:
        path = EXT / f"spearmint_{name}.csv"
        df = pd.read_csv(path)
        sp_pairs = pairs(df, "allele", "peptide_sequence")
        spearmint[name] = {
            "rows": len(df),
            "unique_pairs": len(sp_pairs),
            "eligible_pairs_overlapping": len(elig_pairs & sp_pairs),
        }
    report["spearmint"] = spearmint

    # Any eligible pair in ANY spearmint file (checkpoint exposure).
    all_sp = set()
    for name in spearmint:
        df = pd.read_csv(EXT / f"spearmint_{name}.csv")
        all_sp |= pairs(df, "allele", "peptide_sequence")
    report["eligible_pairs_in_any_spearmint_file"] = len(elig_pairs & all_sp)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "overlap_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
