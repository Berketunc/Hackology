"""Reconcile the revised audit outputs against records_for_review 2.csv.

Read-only check: loads the filtered records with dtype=str, recomputes the
counts reported in audit_summary 2.json, and prints a pass/fail table.
Does not modify any input file.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DOWNLOADS = Path("/Users/berketunc/Downloads")
RECORDS = DOWNLOADS / "records_for_review 2.csv"
SUMMARY = DOWNLOADS / "audit_summary 2.json"

KNOWN_COMMENT = (
    "pMHC-I complex stability was determined by a scintillation "
    "proximity based pMHC-I dissociation assay (PMID 21044632). "
    "The half life reported is an average of at least two "
    "independent experiments."
)


def main():
    data = pd.read_csv(RECORDS, dtype=str).fillna("")
    summary = json.loads(SUMMARY.read_text())

    values = pd.to_numeric(data["half_life_hours_raw"], errors="coerce")
    checks = {}

    checks["human_hla_i_half_life_rows"] = len(data)
    checks["numerical_values_with_known_units"] = int(values.notna().sum())
    checks["zero_values"] = int(values.eq(0).sum())
    checks["known_replicate_comment_rows"] = int(
        data["comments"].eq(KNOWN_COMMENT).sum()
    )
    checks["comments_flagged"] = int(
        data["comment_needs_review"].str.lower().eq("true").sum()
    )
    checks["no_automatic_flags_not_yet_verified"] = int(
        data["flags"].eq("").sum()
    )
    known_positive = (
        data["comments"].eq(KNOWN_COMMENT)
        & data["flags"].eq("")
        & values.gt(0)
    )
    checks["known_replicate_comment_positive_candidates"] = int(
        known_positive.sum()
    )
    checks["duplicate_assay_id_rows"] = int(
        data.duplicated("assay_id", keep=False).sum()
    )
    checks["units"] = data["units"].value_counts().to_dict()
    checks["assay_counts"] = data["assay"].value_counts().to_dict()

    print(f"{'check':<55} {'recomputed':>12} {'summary':>12}  ok")
    failures = 0
    for key, recomputed in checks.items():
        expected = summary.get(key, "<missing>")
        ok = recomputed == expected
        failures += not ok
        print(f"{key:<55} {str(recomputed)[:12]:>12} "
              f"{str(expected)[:12]:>12}  {ok}")

    # Candidate subset detail: per-allele and per-reference counts.
    cand = data.loc[known_positive].copy()
    cand["hours"] = values[known_positive]
    print("\nCandidate subset (positive, unflagged, replicate comment):")
    print(f"  rows={len(cand)}  peptides={cand['peptide'].nunique()}  "
          f"alleles={cand['allele'].nunique()}")
    by_allele = (
        cand.groupby(["allele", "reference_id"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    print(by_allele.to_string(index=False))

    print("\nCandidate integrity checks:")
    print("  distinct (allele,peptide,mod,residues) groups:",
          cand.groupby(
              ["allele", "peptide", "modifications", "modified_residues"]
          ).ngroups)
    print("  conflicting numeric labels within same (allele,peptide):",
          int((cand.groupby(["allele", "peptide"])["hours"]
               .nunique() > 1).sum()))
    print("  inequality values:", cand["inequality"].unique().tolist())
    print("  assay values:", cand["assay"].unique().tolist())
    print("  paper_title values:", cand["paper_title"].unique().tolist())
    print("  modifications nonempty:",
          int((cand["modifications"] != "").sum()),
          " modified_residues nonempty:",
          int((cand["modified_residues"] != "").sum()))
    print("  canonical peptides:",
          int(cand["peptide"].str.fullmatch(
              r"[ACDEFGHIKLMNPQRSTVWY]+").sum()))
    print("  peptide lengths:",
          sorted(cand["peptide"].str.len().unique().tolist()))
    print("  half-life hours range: "
          f"{cand['hours'].min()} .. {cand['hours'].max()}")

    flagged = data.loc[~known_positive]
    print("\nOutside candidate subset:")
    print(f"  rows={len(flagged)}")
    print(flagged["flags"].value_counts().head(20).to_string())

    if failures:
        print(f"\n{failures} RECONCILIATION FAILURE(S)")
        sys.exit(1)
    print("\nAll recomputed counts match audit_summary 2.json.")


if __name__ == "__main__":
    main()
