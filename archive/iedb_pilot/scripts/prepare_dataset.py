"""Prepare the reviewed eligible dataset from records_for_review 2.csv.

Eligibility rule (batch scope, evidence documented in
reports/source_review.md):
  - comments == the known SPA replicate-count comment (PMID 21044632)
  - no automatic flags
  - half_life_hours_raw > 0

Eligible rows are batch-approved at submission level: every one of the 10
IEDB submission references carries the identical abstract stating the data
were generated with the SPA assay of PMID 21044632, whose methods measure
dissociation at 37C. temperature_C=37.0 is therefore PROTOCOL-DERIVED, not
per-row recorded. See reports/source_review.md for the unresolved
assumption (whether each submission followed the protocol unmodified).

Outputs:
  data/processed/eligible_records.csv  - approved candidate rows
  data/processed/excluded_records.csv  - all other rows with reasons
  data/manifests/dataset_manifest.json - counts, checksums, provenance
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.config import ROOT as PROJECT_ROOT
ROOT = PROJECT_ROOT / 'archive' / 'iedb_pilot'
DATA, RESULTS, REPORTS = ROOT / 'data', ROOT / 'results', ROOT / 'reports'

import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
WORKSPACE = ROOT
RECORDS = DOWNLOADS / "records_for_review 2.csv"

KNOWN_COMMENT = (
    "pMHC-I complex stability was determined by a scintillation "
    "proximity based pMHC-I dissociation assay (PMID 21044632). "
    "The half life reported is an average of at least two "
    "independent experiments."
)

PROTOCOL_ID = "SPA_PMID21044632_37C"
TEMPERATURE_C = 37.0
REVIEW_EVIDENCE = (
    "Batch approval, submission-level: all 10 IEDB references "
    "(1028282..1028294) carry the identical abstract 'data generated "
    "using a scintillation proximity assay based peptide-HLA-I "
    "dissociation assay (PMID: 21044632)' (fetched 2026-10-03). "
    "PMID 21044632 section 2.5: dissociation initiated by raising "
    "temperature to 37C and monitored on a 37C-modified TopCount; "
    "18C applies to the overnight refolding/association stage only. "
    "Temperature is protocol-derived; unmodified-protocol adherence "
    "per submission assumed, not per-row verified."
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    data = pd.read_csv(RECORDS, dtype=str).fillna("")
    hours = pd.to_numeric(data["half_life_hours_raw"], errors="coerce")

    eligible = (
        data["comments"].eq(KNOWN_COMMENT)
        & data["flags"].eq("")
        & hours.gt(0)
    )

    out_dir = WORKSPACE / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = data.loc[eligible].copy()
    kept["approved"] = True
    kept["temperature_C"] = TEMPERATURE_C
    kept["protocol_id"] = PROTOCOL_ID
    kept["half_life_hours"] = kept["half_life_hours_raw"]
    kept["review_evidence"] = REVIEW_EVIDENCE

    dropped = data.loc[~eligible].copy()
    dropped["exclusion_reason"] = dropped["flags"].where(
        dropped["flags"].ne(""),
        "not_in_candidate_scope",
    )
    zero_mask = pd.to_numeric(
        dropped["half_life_hours_raw"], errors="coerce"
    ).eq(0)
    dropped.loc[
        zero_mask & dropped["flags"].str.contains("zero"),
        "exclusion_reason",
    ] = dropped.loc[
        zero_mask & dropped["flags"].str.contains("zero"),
        "exclusion_reason",
    ] + " (zero half-life: detection/reporting convention unresolved)"

    kept.to_csv(out_dir / "eligible_records.csv", index=False)
    dropped.to_csv(out_dir / "excluded_records.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "records_for_review_2_csv": {
                "path": str(RECORDS),
                "sha256": sha256(RECORDS),
                "rows": int(len(data)),
            },
            "raw_iedb_export": {
                "path": str(WORKSPACE / "mhc_ligand_full.csv"),
                "sha256": (
                    "a480284fccc7b7af05bf91121ba32718d289fe6394f777525809"
                    "423a0a050a30"
                ),
                "note": "two-header IEDB export; not re-ingested here",
            },
        },
        "eligibility_rule": {
            "comment": "exact SPA replicate-count comment",
            "flags": "empty",
            "half_life": "> 0 hours",
            "protocol_id": PROTOCOL_ID,
            "temperature_C": TEMPERATURE_C,
            "temperature_provenance": "protocol-derived (PMID 21044632)",
        },
        "counts": {
            "eligible_rows": int(len(kept)),
            "eligible_peptides": int(kept["peptide"].nunique()),
            "eligible_alleles": int(kept["allele"].nunique()),
            "excluded_rows": int(len(dropped)),
        },
        "outputs": {
            "eligible_records_csv": "data/processed/eligible_records.csv",
            "excluded_records_csv": "data/processed/excluded_records.csv",
        },
    }
    manifest_path = WORKSPACE / "data" / "manifests" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"eligible: {len(kept)} rows, {kept['peptide'].nunique()} peptides, "
          f"{kept['allele'].nunique()} alleles")
    print(f"excluded: {len(dropped)} rows")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
