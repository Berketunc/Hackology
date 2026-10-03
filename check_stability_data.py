import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


# Matches the two-level headers in your IEDB export.
FIELDS = {
    ("Assay ID", "IEDB IRI"): "assay_id",
    ("Reference", "IEDB IRI"): "reference_id",
    ("Reference", "PMID"): "pmid",
    ("Reference", "Title"): "paper_title",
    ("Epitope", "Name"): "peptide",
    ("Epitope", "Modifications"): "modifications",
    ("Epitope", "Modified residues"): "modified_residues",
    ("MHC Restriction", "Name"): "allele",
    ("MHC Restriction", "Class"): "mhc_class",
    ("Assay", "Method"): "assay",
    ("Assay", "Response measured"): "response",
    ("Assay", "Units"): "units",
    ("Assay", "Quantitative measurement"): "measurement_raw",
    ("Assay", "Measurement Inequality"): "inequality",
    ("Assay", "Comments"): "comments",
}

# Conservative screening flags, not automatic interpretation of comments.
COMMENT_FLAG = re.compile(
    r"[<>≤≥]"
    r"|\b(?:range|between|at least|at most|less than|more than)\b",
    re.IGNORECASE,
)

CANONICAL_PEPTIDE = r"[ACDEFGHIKLMNPQRSTVWY]+"
EXACT_HLA = r"HLA-[ABC]\*\d{2,3}:\d{2,3}(?::\d{2,3})*[A-Z]?"


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_iedb(path):
    """Stream the large CSV; retain only human HLA-I half-life rows."""
    records = []
    total = malformed = 0

    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        top = next(reader)
        bottom = next(reader)

        if len(top) != len(bottom):
            raise ValueError("The two header rows have different widths.")

        headers = [(a.strip(), b.strip()) for a, b in zip(top, bottom)]
        missing = [key for key in FIELDS if key not in headers]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        positions = {
            output: headers.index(key)
            for key, output in FIELDS.items()
        }

        for row in reader:
            if not row or not any(row):
                continue

            total += 1
            if len(row) != len(headers):
                malformed += 1
                continue

            response = row[positions["response"]].strip().lower()
            allele = row[positions["allele"]].strip()
            mhc_class = row[positions["mhc_class"]].strip().upper()

            if (
                re.search(r"half[\s-]*life", response)
                and allele.startswith("HLA-")
                and mhc_class == "I"
            ):
                records.append({
                    name: row[index].strip()
                    for name, index in positions.items()
                })

            if total % 1_000_000 == 0:
                print(f"Read {total:,} rows; retained {len(records):,}")

    return pd.DataFrame(records), total, malformed


def audit(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, total, malformed = read_iedb(args.input)
    if data.empty:
        raise ValueError("No human HLA-I half-life records found.")

    values = pd.to_numeric(data["measurement_raw"], errors="coerce")
    factors = data["units"].str.lower().map({
        "min": 1 / 60,
        "minute": 1 / 60,
        "minutes": 1 / 60,
        "h": 1,
        "hr": 1,
        "hrs": 1,
        "hour": 1,
        "hours": 1,
        "s": 1 / 3600,
        "sec": 1 / 3600,
        "seconds": 1 / 3600,
    })

    data["half_life_hours_raw"] = values * factors
    data["comment_needs_review"] = data["comments"].str.contains(
        COMMENT_FLAG, na=False
    )
    data["canonical_peptide"] = data["peptide"].str.fullmatch(
        CANONICAL_PEPTIDE, na=False
    )
    data["resolved_standard_hla"] = data["allele"].str.fullmatch(
        EXACT_HLA, na=False
    )
    data["has_modification"] = (
        data["modifications"].ne("")
        | data["modified_residues"].ne("")
    )

    def flags(row):
        problems = []
        value = row["half_life_hours_raw"]

        if pd.isna(value):
            problems.append("missing_number_or_unknown_unit")
        elif value < 0:
            problems.append("negative_value")
        elif value == 0:
            problems.append("zero_needs_interpretation")

        if row["inequality"] != "=":
            problems.append("not_explicitly_exact")
        if row["comment_needs_review"]:
            problems.append("comment_may_describe_bound_or_range")
        if not row["canonical_peptide"]:
            problems.append("noncanonical_peptide")
        if not row["resolved_standard_hla"]:
            problems.append("allele_needs_review")
        if row["has_modification"]:
            problems.append("modified_peptide")

        return ";".join(problems)

    data["flags"] = data.apply(flags, axis=1)

    # Blank fields are intentional. Do not invent temperature or approval.
    data["approved"] = False
    data["temperature_C"] = np.nan
    data["protocol_id"] = ""
    data["half_life_hours"] = data["half_life_hours_raw"]
    data["review_evidence"] = ""

    data.to_csv(out / "records_for_review.csv", index=False)

    (
        data.groupby(["assay", "reference_id", "pmid"], dropna=False)
        .agg(
            rows=("assay_id", "size"),
            numerical_rows=("half_life_hours_raw", "count"),
            peptides=("peptide", "nunique"),
            alleles=("allele", "nunique"),
        )
        .reset_index()
        .to_csv(out / "counts_by_assay_and_study.csv", index=False)
    )

    numeric = data.loc[data["half_life_hours_raw"].notna()].copy()
    pair_columns = [
        "allele", "peptide", "modifications", "modified_residues"
    ]
    method_counts = numeric.groupby(pair_columns)["assay"].nunique()

    duplicates = data[data.duplicated("assay_id", keep=False)]
    duplicates.to_csv(out / "duplicate_assay_ids.csv", index=False)

    report = {
        "input_rows": total,
        "malformed_rows_skipped": malformed,
        "human_hla_i_half_life_rows": len(data),
        "numerical_values_with_known_units": int(
            data["half_life_hours_raw"].notna().sum()
        ),
        "zero_values": int(data["half_life_hours_raw"].eq(0).sum()),
        "comments_flagged": int(data["comment_needs_review"].sum()),
        "no_automatic_flags_not_yet_verified": int(
            data["flags"].eq("").sum()
        ),
        "distinct_pair_modification_groups": len(method_counts),
        "groups_with_multiple_methods": int(method_counts.gt(1).sum()),
        "assay_counts": data["assay"].value_counts().to_dict(),
        "units": data["units"].value_counts(dropna=False).to_dict(),
        "temperature_verified": False,
        "checkpoint_training_overlap_audited": False,
        "sufficiency_verdict": "NOT YET ESTABLISHED",
    }

    save_json(out / "audit_summary.json", report)
    print(json.dumps(report, indent=2))
    print(f"\nReview file: {out / 'records_for_review.csv'}")


def features(data):
    """Simple baseline: allele identity + positional peptide encoding."""
    result = []
    for row in data.itertuples():
        item = {
            f"allele={row.allele}": 1.0,
            f"length={len(row.peptide)}": 1.0,
        }
        for position, amino_acid in enumerate(row.peptide):
            item[f"position={position}:aa={amino_acid}"] = 1.0
        result.append(item)
    return result


def pilot(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, dtype=str).fillna("")

    required = {
        "approved", "temperature_C", "protocol_id", "half_life_hours",
        "review_evidence", "assay", "allele", "peptide",
        "modifications", "modified_residues", "assay_id",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {required - set(data.columns)}")

    approved = data["approved"].str.lower().isin(["true", "yes", "1"])
    data["temperature_C"] = pd.to_numeric(
        data["temperature_C"], errors="coerce"
    )
    data["half_life_hours"] = pd.to_numeric(
        data["half_life_hours"], errors="coerce"
    )

    selected = (
        approved
        & data["assay"].eq(args.assay)
        & data["temperature_C"].eq(args.temperature)
        & data["protocol_id"].eq(args.protocol)
    )
    data = data.loc[selected].copy()

    if data.empty:
        raise ValueError("No approved rows match the requested conditions.")

    if data["review_evidence"].str.strip().eq("").any():
        raise ValueError("Approved rows must have source-review evidence.")

    if not (
        np.isfinite(data["half_life_hours"])
        & data["half_life_hours"].ge(0)
    ).all():
        raise ValueError("Approved half-lives must be finite and nonnegative.")

    # Deliberately narrow first baseline.
    supported = (
        data["peptide"].str.fullmatch(CANONICAL_PEPTIDE, na=False)
        & data["allele"].str.fullmatch(EXACT_HLA, na=False)
        & data["modifications"].eq("")
        & data["modified_residues"].eq("")
    )
    if not supported.all():
        raise ValueError(
            "The pilot currently supports unmodified canonical peptides "
            "and resolved HLA-A/B/C alleles only."
        )

    data = data.drop_duplicates("assay_id")

    # Do not silently average conflicting measurements.
    keys = ["allele", "peptide"]
    counts = data.groupby(keys)["half_life_hours"].transform("nunique")
    conflicts = data.loc[counts > 1]
    conflicts.to_csv(out / "conflicting_measurements.csv", index=False)
    data = data.loc[counts == 1].copy()

    # Identical repeated pair outcomes do not become extra label purchases.
    # Keep their provenance in a separate file.
    data.to_csv(out / "eligible_measurement_records.csv", index=False)
    data = data.drop_duplicates(keys).reset_index(drop=True)

    groups = data["peptide"]

    # This is a technical guard, NOT a scientific sufficiency threshold.
    if groups.nunique() < 30:
        raise ValueError(
            "Fewer than 30 unique peptides remain. Too few for this pilot "
            "configuration; inspect exclusions and reconsider the design."
        )

    # Hold back a final test set. The pilot does not evaluate on it.
    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=2026
    )
    remaining, final_test = next(outer.split(data, groups=groups))
    remaining_data = data.iloc[remaining]

    # Of remaining groups, 25% = approximately 20% of all groups.
    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=2027
    )
    pool_local, dev_local = next(inner.split(
        remaining_data, groups=remaining_data["peptide"]
    ))
    pool = remaining[pool_local]
    development = remaining[dev_local]

    for left, right in [
        (pool, development), (pool, final_test), (development, final_test)
    ]:
        assert not (
            set(data.iloc[left]["peptide"])
            & set(data.iloc[right]["peptide"])
        )

    data["partition"] = ""
    data.loc[pool, "partition"] = "acquisition_pool"
    data.loc[development, "partition"] = "development"
    data.loc[final_test, "partition"] = "final_test"
    data.to_csv(out / "partitioned_data.csv", index=False)

    # Fit preprocessing on pool inputs only.
    # Pool labels remain hidden except for each selected training subset.
    vectorizer = DictVectorizer(sparse=True)
    x_pool = vectorizer.fit_transform(features(data.iloc[pool]))
    x_dev = vectorizer.transform(features(data.iloc[development]))
    y_pool = np.log1p(data.iloc[pool]["half_life_hours"].to_numpy())
    y_dev = np.log1p(
        data.iloc[development]["half_life_hours"].to_numpy()
    )

    n = len(pool)
    budgets = sorted(set(
        max(2, min(n, int(round(n * fraction))))
        for fraction in [0.05, 0.10, 0.25, 0.50, 1.0]
    ))

    results = []
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(n)

        for budget in budgets:
            chosen = order[:budget]

            model = Ridge(alpha=10.0, solver="lsqr")
            model.fit(x_pool[chosen], y_pool[chosen])
            prediction = model.predict(x_dev)

            constant = np.full(len(y_dev), y_pool[chosen].mean())

            results.append({
                "seed": seed,
                "labels_acquired": budget,
                "rmse_log1p": float(np.sqrt(
                    mean_squared_error(y_dev, prediction)
                )),
                "mae_log1p": float(
                    mean_absolute_error(y_dev, prediction)
                ),
                "constant_rmse_log1p": float(np.sqrt(
                    mean_squared_error(y_dev, constant)
                )),
            })

    results = pd.DataFrame(results)
    results.to_csv(out / "pilot_runs.csv", index=False)

    summary = results.groupby("labels_acquired").agg(
        rmse_mean=("rmse_log1p", "mean"),
        rmse_sd=("rmse_log1p", "std"),
        mae_mean=("mae_log1p", "mean"),
        constant_rmse_mean=("constant_rmse_log1p", "mean"),
    )
    summary.to_csv(out / "pilot_summary.csv")

    print("\nPartitions:")
    print(data.groupby("partition").agg(
        pairs=("assay_id", "size"),
        peptides=("peptide", "nunique"),
        alleles=("allele", "nunique"),
    ).to_string())

    print("\nDevelopment learning curve:")
    print(summary.to_string())
    print("\nFinal-test performance was NOT evaluated.")
    print("Exact peptide overlap is prevented; near-neighbour overlap is not.")
    print("This pilot has no pretrained backbone and does not audit one.")
    print("Learning-curve results do not automatically establish sufficiency.")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("input")
    audit_parser.add_argument("--out", default="stability_audit")
    audit_parser.set_defaults(function=audit)

    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("input")
    pilot_parser.add_argument("--assay", required=True)
    pilot_parser.add_argument("--temperature", type=float, required=True)
    pilot_parser.add_argument("--protocol", required=True)
    pilot_parser.add_argument("--out", default="stability_pilot")
    pilot_parser.set_defaults(function=pilot)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()