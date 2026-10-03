# Archived IEDB active-learning pilot

This is the earlier **5,815 positive SPA pairs / 10 alleles** study, not the challenge benchmark. Its peptide lengths are 8–13, zeros were excluded, and its final test remains untouched. The challenge benchmark instead uses the supplied Rasmussen 9-mer table with zeros retained; see the root [README](../../README.md).

Original reports are in [reports/](reports/); `results/pilot_all_alleles/partitioned_data.csv` retains the original 3,457 / 1,188 / 1,170 pool/development/test assignment. Data and result contents were moved without recomputing them. Paths in historical reports/manifests describe their original locations.

Run archived scripts from the repository root as modules, for example `python -m archive.iedb_pilot.scripts.predictor_comparison`. Their output roots now point into this archive. Shared original external source CSVs remain under root `data/external/`; the large raw IEDB export remains uncommitted. The previously tracked 19 MB pilot embedding cache is retained here; it is distinct from the uncommitted main-benchmark feature arrays.
