# Supplied-data provenance receipt

The user-provided [checkv1.md](checkv1.md), section 4, reports an independent direct download of the organizer spreadsheet and a content comparison after sorting against `data/processed/rasmussen_all.csv`. It reports one tab (`gid=1629690744`), 28,166 rows × five source columns, 75 allele labels, nine-residue peptides, 34-residue pseudosequences, 182-residue `hla_seq`, three C67S labels, no duplicate allele–peptide pairs, and no HLA-C rows.

This is a retained **user-supplied verification receipt**, not a claim that the current agent repeated that live-sheet comparison. Local provenance and input invariants are recorded in [rasmussen_manifest.json](../data/manifests/rasmussen_manifest.json) and enforced by [src/data.py](../src/data.py). The processed CSV additionally carries derived columns and a row index. Its 5,679 zero labels comprise about 20.2% of the original 28,166 rows.

The distinct 5,815-positive-pair IEDB pilot is archived under [archive/iedb_pilot](../archive/iedb_pilot/). Its partitions and conflicting-measurement files are not the challenge benchmark or replicate evidence for it.
