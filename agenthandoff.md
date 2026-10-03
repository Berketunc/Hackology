# Agent handoff: active learning for peptide–HLA stability

Updated: 3 October 2026 (session 2: source review, pilots, embeddings, and the three-policy comparison are DONE)  
Workspace: /Users/berketunc/Hackology

## Start here: current handoff to Codex CLI

**Session 2 completed the full pipeline through the three-policy active-learning comparison.** Everything below remains accurate as background; the new state is summarised in Section 0 and detailed in `reports/al_comparison.md`.

Headline result (development set only, final test untouched): **random acquisition beat both sequence-diversity and embedding-diversity at every budget in all 10 seeds** — a clean negative result for pure diversity acquisition with this predictor. Under an identical Ridge fitting procedure, one-hot positional features also beat mean-pooled ESM-2 pair features (0.912 vs ~1.11 at 2,076 labels; ~0.93 with proper regularisation). The result is robust to a sequence-similarity sensitivity analysis.

Natural next step before any demo: test an **uncertainty-based acquisition policy** (bootstrap-ensemble or GP regression on the same frozen ESM-2 features, same partitions, same budgets) to separate "diversity doesn't help" from "embeddings don't help". Final-test evaluation stays reserved for a locked design.

Use the four files with " 2" in their names in Section 4 as the current inputs. Preserve both versions in Downloads. Write code, reviewed derivatives, provenance notes, manifests, and results under this workspace in clearly named directories. Do not mark all rows approved or fill missing temperature merely to make the pilot run.

The corrected audit leaves 6,101 rows with no automatic flags. The main candidate pool is the 5,815 positive records with the verified replicate-count comment, covering 4,268 peptides and 10 alleles. These are candidates, not yet fully source-verified labels. The smaller HLA-A*02:01 group has 977 positive records and can serve as an initial within-allele pilot.

New source evidence: the cited method paper, PMID 21044632, explicitly describes initiating dissociation at 37°C. Use https://pmc.ncbi.nlm.nih.gov/articles/PMC4341823/ and its methods as evidence. Distinguish this published protocol fact from the remaining question of whether each IEDB submission followed that protocol without changes. Record the source and inference level of any temperature annotation.

Session 2 (see Section 0): the pilots, embedding extraction, predictor comparison, and three-policy comparison have all been run; a source-reviewed eligible dataset exists. The final test set has still never been evaluated, and no model improvement over simple baselines has been demonstrated — the diversity-policy result was negative.

## 0. Session 2 state (what now exists)

### Verified and prepared

- **Reconciliation**: `scripts/reconcile_audit.py` confirms every count in `audit_summary 2.json` recomputes exactly from `records_for_review 2.csv` (10,605 rows; 6,101 unflagged; 5,815 candidates; 0 duplicate assay IDs).
- **Source verification (new evidence)**: all 10 IEDB submission references (1028282–1028294) were fetched live on 2026-10-03; each carries the identical abstract stating data were "generated using a scintillation proximity assay based peptide-HLA-I dissociation assay (PMID: 21044632)". PMC4341823 §2.5 was re-verified: refold at 18 °C, dissociation initiated at 37 °C on a 37 °C-modified TopCount. Evidence chain in `reports/source_review.md`.
- **Eligible dataset**: `data/processed/eligible_records.csv` — 5,815 rows batch-approved at submission level, `temperature_C=37.0` marked **protocol-derived**, `protocol_id=SPA_PMID21044632_37C`, per-row `review_evidence`. `excluded_records.csv` keeps all 4,790 other rows with reasons. `eligible_hla_a0201.csv` is the 977-row A*02:01 subset.
- **Manifests**: `data/manifests/input_manifest.json` + `dataset_manifest.json` — SHA-256 for all inputs incl. `mhc_ligand_full.csv` (`a480284f…50a30`, confirmed two-header raw IEDB export). Volatile /tmp snapshots promoted to `data/external/` (Rasmussen sheet + 6 SPEARMINT split files) with `SHA256SUMS.txt`.
- **Overlap audit** (`reports/overlap_report.json`): 5,517/5,815 pairs match the Rasmussen sheet; 5,490 exactly equal in hours, rest within 0.003 h; the 298 unmatched are all non-9-mers (sheet is 9-mers only). **5,807/5,815 eligible pairs appear in SPEARMINT files** (4,420 in its stability-train) — a SPEARMINT stability checkpoint may never supply embeddings here.
- **Unresolved assumptions** (documented, still open): protocol adherence assumed not per-row verified; 482 zeros excluded (population = positive recorded half-lives); labels are ≥2-replicate means; binder-enriched pool; near-sequence grouping only exact.

### Results

- **Pilots** (dev only): A*02:01 RMSE 1.066→0.846 vs ~1.10 constant (`results/pilot_a0201/`); 10-allele 1.017→0.898 vs ~1.07 (`results/pilot_all_alleles/`). Partitions live in `results/pilot_all_alleles/partitioned_data.csv` (pool 3,457 / dev 1,188 / final test 1,170) — **reuse these; final test was never evaluated**.
- **Embeddings**: `facebook/esm2_t33_650M_UR50D`, frozen, mean-pool last hidden layer; pair feature = concat(peptide, mhc full-length heavy chain from SPEARMINT files; canonical A*02:01 variant). Generated once on Modal T4 (~24 s, ~$0.01) via `scripts/modal_embed.py`; cached at `data/processed/embeddings_esm2.npz` + meta.
- **Predictor comparison** (`results/predictor_comparison/`): one-hot + Ridge(alpha=10) beat ESM-2 + Ridge(alpha=10) at every budget (0.912 vs 1.115 at 2,076). Diagnostic: alpha=10 is mis-scaled for 2,560-dim embeddings; `RidgeCV(LOO)` recovers to ~0.93 — still slightly behind one-hot. Mean-pooling plausibly erases positional anchor signal.
- **Three-policy comparison** (`results/policy_comparison/`, design in `design.json`): identical `RidgeCV` on standardized ESM-2 pair features; matched init 173, batch 173, budgets →2,076; 10 seeds. **Negative**: AULC random 1.073, embdiv 1.086, seqdiv 1.089; random best at every budget; diversity won 0/10 seeds.
- **Sensitivity** (`results/sensitivity/`): only 2.0% of dev peptides have a ≥90%-identity pool neighbour; stratified dev RMSE (<70–80% identity to acquired set) leaves ranking unchanged — not leakage-driven.

### Environment

- `.venv` (uv): pandas 3.0.6, sklearn 1.9.1, numpy 2.4.6, torch 2.14.1, transformers 5.18.0, modal. Run scripts with `.venv/bin/python`.
- Modal: authenticated, workspace `tnberkec`, payment method on file; GPU works. `scripts/modal_embed.py` has `::benchmark` and `::full` entrypoints (`--device gpu|cpu`).
- HF_TOKEN lives in `~/.env` (read by `modal_embed.py`).

### Open next steps (suggested, not started)

1. Uncertainty policy: bootstrap-ensemble or GP regression on the same ESM-2 features, identical partitions/budgets — distinguishes "diversity fails" from "ESM-2 features fail".
2. Density-weighted diversity (cluster-then-uncertainty) to fix the outlier-grabbing failure.
3. Alternative pair representations (e.g., per-residue pooling, ESM-2 on peptide+MHC complex) — one-hot already beats current features, so predictor strength bounds what any policy can show.
4. Final-test evaluation ONLY after the acquisition design is locked on dev.
5. Optional demo after science (`modal_embed.py` shows the Modal pattern).

---

## 1. Purpose and agreed direction

Build a retrospective active-learning experiment for peptide–HLA class I complex stability. The user is participating in the AI × Science hackathon, Track 3: Drug and protein design by Serova. They want implementation and evidence, not repeated high-level planning.

The user does not have a biology background. Explain decisions in plain language and distinguish verified facts, hypotheses, and unresolved questions. They explicitly asked for an honest sufficiency assessment without hidden assumptions.

The current project asks:

> Can protein-model representations select peptide–HLA stability measurements that improve prediction faster than random or sequence-based selection?

Keep the measurement method fixed for the first experiment. The earlier proposal to recommend which assay to run has been deferred because paired cross-assay data are sparse and there is no established reference objective or cost model.

The end product should first be a reproducible scientific pipeline and learning curves. A UI is optional after results exist.

## 2. Minimal biological context

A peptide is a short amino-acid sequence. An HLA molecule is a protein that can bind and display peptides. An HLA allele identifies a particular variant of that protein. A peptide–HLA pair is the experimental object.

The outcome is a measured dissociation half-life under specified conditions. A laboratory assay measures that outcome. Half-life, binding affinity, qualitative binding, melting temperature, and immune response are different endpoints. Do not pool them as interchangeable labels.

A protein embedding is a numerical representation of a sequence or pair. In this experiment the large representation model stays frozen and a small regression predictor learns from the revealed stability labels.

Active learning selects which labels to reveal next. It does not necessarily select the peptides with the highest stability; improving prediction and discovering the best candidates are different objectives.

## 3. What counts as success

At matched label budgets, compare selection strategies using the same predictor, feature representation, fitting procedure, initial labels, acquisition pool, and evaluation set.

Primary comparison:
- Random acquisition.
- Sequence-based diversity acquisition.
- Protein-embedding diversity acquisition.

After a selection batch is revealed, refit the small predictor using all acquired labels. Recalculate acquisition scores relative to the current labelled set. Within a diversity batch, account for points already selected in that batch.

Primary figure: held-out MAE/RMSE in log1p(hours) versus total labels acquired, including the initial labelled set. Repeat across initial seeds and show variability. Report study/allele coverage and the intended target population.

A supported result might be: one policy reaches a prespecified prediction error using fewer retrospective label acquisitions. Do not promise that embedding selection will beat random. Negative results are valid.

Do not claim demonstrated laboratory savings, therapeutic benefit, improved immunogenicity, or a general assay recommender.

## 4. Files currently available

Current revised inputs, inspected read-only after the user reran the corrected audit:

| File | Purpose |
| --- | --- |
| /Users/berketunc/Downloads/records_for_review 2.csv | Current filtered IEDB records, corrected flags, and unfilled review fields |
| /Users/berketunc/Downloads/duplicate_assay_ids 2.csv | Current duplicate assay-ID report; zero data rows |
| /Users/berketunc/Downloads/counts_by_assay_and_study 2.csv | Current counts, including no_automatic_flags |
| /Users/berketunc/Downloads/audit_summary 2.json | Current summary: 6,101 unflagged rows; 5,815 main positive candidates; 6 comments flagged |

The space and "2" are part of each filename. Quote paths in shell commands. Verified review state: zero approved rows and zero populated temperature fields.

Earlier inputs retained for provenance; do not use their original flags as the current result:

| File | Purpose |
| --- | --- |
| /Users/berketunc/Downloads/records_for_review.csv | Filtered IEDB records with original data and review fields |
| /Users/berketunc/Downloads/duplicate_assay_ids.csv | Duplicate assay-ID report; currently zero data rows |
| /Users/berketunc/Downloads/counts_by_assay_and_study.csv | Counts by assay and reference |
| /Users/berketunc/Downloads/audit_summary.json | Original audit summary; its comment-screening counts are stale |

The raw full IEDB export was processed on a teammate's computer. Its current filename/location on this host has not been established. Locate it or ask for its path only if needed for a particular verification. The filtered review CSV already contains the retained original fields and is sufficient to begin source review and overlap analysis. It has a single header row; do not parse it as the raw two-header IEDB file.

The original Python program was supplied in chat as check_stability_data.py. The user's earlier console showed python3 main.py on a teammate's machine. No corresponding Python source file was found in this workspace during the initial handoff. The user subsequently reran the corrected function elsewhere and supplied the revised outputs above. Locate existing project code before editing; if none exists locally, implement a small reproducible local pipeline from the documented schemas and algorithms. Do not let an unavailable original script block analysis of the current filtered CSV.

This side conversation created and updated this handoff artifact. Function validation used temporary outputs. The user's files in Downloads have not been overwritten, and no persistent reviewed dataset or pilot outputs have been created here.

The original Rasmussen Google Sheet:
https://docs.google.com/spreadsheets/d/1NtZNvcF3u0KFn-1bbuA50CF3IXvs1l4HfbaR3KRLvso/edit?gid=1629690744#gid=1629690744

A CSV snapshot was previously exported to:
 /var/folders/8_/zx9x8l493j7b9wgkvs449q880000gn/T/browser-use/exports/rasmussen_et_al_dataset-90425f97-244a-476c-83d0-80cbcf4e1a09.csv

Temporary source snapshots from earlier investigations may exist:
- /private/tmp/spearmint_uq_train.csv
- /private/tmp/spearmint_uq_bs_val.csv
- /private/tmp/spearmint_uq_bs_test.csv
- /private/tmp/spearmint_uq_s3_train.csv
- /private/tmp/spearmint_uq_s3_val.csv
- /private/tmp/spearmint_uq_s3_test.csv

Check existence before relying on temporary files. Record versions/checksums if promoting them into an experiment.

## 5. Verified IEDB audit findings

The original audit processed 5,536,251 records, with no malformed rows skipped.

| Finding | Count |
| --- | ---: |
| Human HLA-I half-life rows | 10,605 |
| Numerical values with known units | 10,489 |
| Zero half-life values | 523 |
| Distinct peptide–HLA/modification groups among numerical rows | 10,176 |
| Groups measured with multiple methods | 62 |
| Duplicate assay-ID rows | 0 |

All filtered rows specify minutes. Convert minutes to hours by division by 60, retaining the original number and units. This normalises units; it does not independently verify that source curation was correct.

Assay counts, before numerical completeness and cleaning:
- purified MHC/direct/radioactivity: 6,780
- purified MHC/direct/fluorescence: 3,428
- cellular MHC/direct/fluorescence: 314
- binding assay: 72
- cellular MHC/direct/radioactivity: 7
- lysate MHC/direct/radioactivity: 4

There is no verified temperature metadata in the current reviewed output. The original audit deliberately leaves temperature_C, protocol_id, and review_evidence blank and approved=False.

## 6. Confirmed defect in the first audit code

The initial comment regular expression flagged "at least" anywhere. It therefore flagged this exact comment in 6,297 records:

"pMHC-I complex stability was determined by a scintillation proximity based pMHC-I dissociation assay (PMID 21044632). The half life reported is an average of at least two independent experiments."

This is a statement about replicate count, not a bound on half-life.

The replacement audit exempts only that complete exact comment from the bound/range text screen. It preserves the original comment and records known_replicate_comment=True. It does not globally remove "at least", clear structured inequalities, approve rows, or infer temperature.

After that correction, verified against the supplied filtered records:

| Finding | Original | Corrected |
| --- | ---: | ---: |
| Comments flagged | 6,303 | 6 |
| Rows with no automatic flags | 286 | 6,101 |
| Positive, otherwise unflagged rows carrying the known replicate comment | — | 5,815 |

Passing a regex screen does not establish scientific validity. The corrected 6,101 remain unapproved.

The replacement function is included in Appendix A for reference and reproducibility. Initial local validation reran the audit logic on filtered records through a test reader. Subsequently the user supplied the revised full-run outputs named with " 2". Their JSON reports 5,536,251 input rows and zero malformed rows; inspection of their filtered records confirmed 6,101 unflagged records, zero approved records, blank temperatures, and zero duplicate assay IDs. Do not rerun the full ingestion merely to repeat this confirmation.

## 7. Strongest candidate subset

Within the 6,297 records carrying the exact replicate comment:
- All structured inequalities are "=".
- 5,815 have positive numerical half-lives.
- 482 are zero and still require interpretation.
- The positive subset contains 5,815 distinct peptide–HLA pairs, 4,268 distinct peptides, and 10 alleles.
- No conflicting numerical outcomes were found for identical peptide–HLA pairs within this subset.
- All are purified MHC/direct/radioactivity records titled "Large scale analysis of peptide - HLA-I stability."

Positive candidate counts:

| Allele | IEDB reference ID | Positive rows |
| --- | --- | ---: |
| HLA-B*15:01 | 1028293 | 1,020 |
| HLA-A*02:01 | 1028285 | 977 |
| HLA-A*03:01 | 1028288 | 698 |
| HLA-B*07:02 | 1028291 | 617 |
| HLA-B*35:01 | 1028292 | 597 |
| HLA-A*24:02 | 1028289 | 591 |
| HLA-A*11:01 | 1028287 | 575 |
| HLA-B*40:01 | 1028294 | 294 |
| HLA-A*26:01 | 1028290 | 261 |
| HLA-A*01:01 | 1028282 | 185 |

Reference IDs are full IRIs in the CSV, e.g. http://www.iedb.org/reference/1028285.

These submission records often have blank PMID fields. They must not be discarded merely for lacking a PMID. The comment cites a protocol paper:
https://pubmed.ncbi.nlm.nih.gov/21044632/

The method paper's dissociation sections explicitly state that temperature was raised to 37°C to initiate dissociation. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC4341823/. This has now been checked. The outstanding task is to establish and document applicability to the ten submission datasets and any protocol differences. If temperature is assigned through the cited protocol, label its provenance as protocol-derived rather than directly recorded per row. Do not treat every temperature mentioned in the paper as the measurement temperature; association and dissociation stages differ.

A small first pilot could use the 977 positive HLA-A*02:01 records. A larger experiment can include all ten alleles after protocol validation. A single-allele pilot supports only within-allele conclusions.

If zeros are excluded, explicitly define the pilot population as positive recorded half-lives. Do not generalise that result to nonbinders or arbitrary candidates.

## 8. Other label-quality issues already confirmed

Do not treat a missing inequality as an exact observation:
- IEDB assay 5865 has numeric 480 minutes and a blank inequality, but its comment states ">8" hours.
- Assay 5881 has the same issue.
- Assay 17334 has numeric 384 minutes, but its comment describes a 6.4–7-hour range.

Preserve exact, lower-bound, upper-bound, interval, and unresolved outcomes separately. Start an ordinary-regression MVP using verified exact outcomes. Censored/interval modelling is optional later.

Other comments can describe approximate values digitised from figures, peptide modifications, or temperature-dependent experiments. The regex is a triage aid, not a comprehensive classifier.

Zero values may represent a detection/reporting convention. Do not transform them into exact known physical zero without source evidence.

## 9. Rasmussen data and checkpoint leakage

The Rasmussen sheet was independently inspected earlier:
- 28,166 rows and unique peptide–HLA pairs.
- 75 HLA variants.
- 5,633 distinct peptides.
- All peptide lengths are nine residues.
- 5,679 zero values (about 20.2%).
- No missing values in its five columns.
- All provided hla_seq strings are 182 residues; pseudosequences are 34 residues.
- Three allele labels include engineered C67S variants, totalling 1,135 rows.

These are not automatically full-length inputs for MINT/SPEARMINT, which document full heavy-chain inputs. Map sequences appropriately and handle engineered variants explicitly.

A prior direct comparison with the published SPEARMINT files found:
- 21,626 Rasmussen rows match the stability-training file on allele, peptide, and numeric half-life.
- 2,705 match the stability-validation file on allele and peptide.
- 2,700 match the stability-test file on allele and peptide.
- 568 of those stability-test pairs also appear in later stage-3 training.
- 2,090 of those stability-test pairs remain after exclusion from both stages' training and validation files.

These historical overlap counts do not certify the remaining rows as independent: affinity-stage exposure, near-sequence overlap, and checkpoint provenance still need review. The new IEDB subset has not yet been directly joined to Rasmussen in this side conversation.

The released SPEARMINT stability-trained checkpoint must not supply embeddings for an experiment that pretends its already-seen stability labels are unavailable. A fresh head does not erase supervised information in the backbone.

Prefer a general pretrained protein model such as ESM-2, documenting the actual checkpoint. Sequence exposure during self-supervised pretraining is not the same as supervised stability-label exposure. An earlier MINT/affinity checkpoint requires a specific provenance audit before use.

## 10. Implementation sequence

### A. Resume from the completed corrected audit

Load records_for_review 2.csv with dtype=str and preserve identifiers. Reconcile its counts once against audit_summary 2.json and establish an immutable input manifest with checksums. Never overwrite the Downloads files. A fresh source import is necessary only when the retained fields do not answer a specific question.

The full raw IEDB export has two header rows. The current filtered records_for_review 2.csv has one. The original read_iedb() streams the full export with Python csv.reader; Appendix A's audit() expects that reader. A preparation step operating on the current filtered CSV should use the single-header schema directly.

Use bracket access for columns such as data["flags"]. data.flags is a pandas property, not this dataset's flags column.

### B. Verify source conditions and resolve labels

Prioritise the ten submission references covering the 5,815 positive records. Inspect source records, protocols, supplementary tables, and related publications. Record temperature, protocol details, units, definition of outcome, and zero/censoring conventions.

Populate approved, temperature_C, protocol_id, half_life_hours, and review_evidence only when supported. Source-based batch approval is acceptable for homogeneous records if the scope and evidence are explicit. Do not invent conditions or approve records merely because flags is empty.

If exact conditions cannot be recovered, report the concrete missing evidence and either narrow the subset or label the analysis exploratory.

### C. Audit duplication, overlap, and sequence input

Join datasets on normalised allele, peptide, modifications, and relevant condition. Compare numeric labels only after unit normalisation. Preserve disagreements and provenance; do not silently average conflicting measurements.

Check the chosen model's training history. Keep engineered alleles and modified peptides out of the first simple baseline unless represented correctly.

Confirm whether paired row counts and unique-peptide counts are sufficient for a meaningful split. No universal row threshold proves sufficiency.

### D. Create evaluation partitions

Use a fixed acquisition pool, a development set for method/hyperparameter choices, and a final test set untouched until choices are locked.

At minimum, group all occurrences of a peptide together across alleles. Prefer similarity-aware grouping after inspecting near neighbours. Group by study as appropriate for the intended generalisation claim. Document when one-allele-per-submission structure makes study-held-out evaluation effectively allele-held-out evaluation.

Use the same partitions for every acquisition method and record all seeds. Count label budgets in pair-level measurements, not unique peptide counts. Disclose development-label use.

### E. Cheap pilot before embeddings

The existing pilot in chat uses positional peptide one-hot features, allele identity, and a Ridge regression model predicting log1p(hours). It compares predictions with a training-mean constant baseline.

It creates approximately 60% acquisition-pool, 20% development, and 20% final-test peptide groups, with fixed seeds. It evaluates only on development data at nested random label budgets over five seeds. It does not evaluate the final test set.

The implementation's minimum-30-peptide guard is only a technical guard, not a sufficient sample-size claim. Exact peptide grouping does not prevent near-sequence leakage. Its regularisation choice is a starting default, not evidence of an optimal model.

A weak pilot result does not mathematically prove a protein model cannot work. It is a low-cost diagnostic for label quality, target variation, and tractable learning.

### F. Frozen representations and active learning

Verify runtime and GPU access, extract embeddings once, and cache them with checkpoint/version, sequence-processing, and pooling metadata. Include peptide and HLA information; specify how separate representations are combined if using ESM-2.

For the main policy comparison, use the same representation-based predictor for every policy. Random and sequence-based selection must not be paired with a different downstream predictor.

Initially compare:
1. Random selection.
2. Sequence-based diversity.
3. Embedding-based diversity, accounting for the current labelled set and the current batch.

Use common initial labels and budgets across policies. Fit the small model on all acquired labels each round. Never use hidden pool outcomes in scoring or normalisation. Refit label-dependent components only on acquired labels; document any unsupervised use of pool inputs.

Choose parameters using development runs, then lock methods. Evaluate final learning curves once the experimental design is fixed. Show uncertainty across seeds and relevant allele-level results.

### G. Report honest conclusions

Report counts retained/excluded and reasons, the source population, model provenance, split rules, budgets, baseline comparisons, and limitations.

Success is evidence of improved retrospective label efficiency, not proof of laboratory cost savings or clinical utility. A demo is optional after the science works.

## 11. Deliverables status

All items below are DONE as of session 2 (2026-10-03):

1. ~~Source-supported review~~ → `reports/source_review.md`; all 10 reference pages fetched, identical SPA abstract; 37 °C protocol-derived.
2. ~~Preparation code + manifest~~ → `scripts/prepare_dataset.py`, `reconcile_audit.py`; `data/manifests/`.
3. ~~Overlap/provenance report~~ → `scripts/overlap_report.py`, `reports/overlap_report.json`. SPEARMINT stability checkpoint confirmed unusable.
4. ~~Reviewed dataset + split manifest~~ → `data/processed/eligible_records.csv`, `excluded_records.csv`; `results/*/partitioned_data.csv`.
5. ~~Cheap dev learning curves~~ → `results/pilot_a0201/`, `results/pilot_all_alleles/`; both beat constant baseline.
6. ~~Proceed/narrow decision~~ → PROCEED was taken; AL comparison ran.
7. ~~Embeddings + three-policy comparison~~ → `results/policy_comparison/`, `reports/al_comparison.md`. **Negative result**: random beat both diversity policies everywhere.

Remaining / next (see Section 0 "Open next steps"): uncertainty- or density-weighted acquisition, possibly a better pair representation, then a locked-design final-test evaluation. A demo is optional after the science.

## 12. Useful sources

- IEDB: https://www.iedb.org/
- IEDB query API: https://query-api.iedb.org/docs/swagger/
- SPA method paper: https://pubmed.ncbi.nlm.nih.gov/21044632/
- SPA method full text, including 37°C dissociation procedure: https://pmc.ncbi.nlm.nih.gov/articles/PMC4341823/
- Original NetMHCstabpan study: https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/
- SPEARMINT repository: https://github.com/dhuvik/spearmint
- Stability splits: https://github.com/dhuvik/spearmint/tree/main/data/binding_stability
- Multi-assay splits: https://github.com/dhuvik/spearmint/tree/main/data/stage3_assay_conditioning
- SPEARMINT model card: https://huggingface.co/dkarthikeyan1/spearmint
- Protein uncertainty/active-learning benchmark: https://pmc.ncbi.nlm.nih.gov/articles/PMC11741572/

## Appendix A. Replacement audit() function

Drop-in replacement for the function in the script previously supplied in chat. Retain its imports (Path, pandas as pd, numpy as np, json), read_iedb(), save_json(), COMMENT_FLAG, CANONICAL_PEPTIDE, and EXACT_HLA definitions.

The function resets review fields intentionally. It has already been run by the user to produce the revised inputs. If a rerun becomes necessary, use raw input and a new audit directory; never overwrite a manually reviewed file with regenerated defaults.

```python
def audit(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, total, malformed = read_iedb(args.input)
    if data.empty:
        raise ValueError("No human HLA-I half-life records found.")
    data = data.fillna("").copy()

    values = pd.to_numeric(data["measurement_raw"], errors="coerce")
    factors = data["units"].str.lower().map({
        "min": 1 / 60, "minute": 1 / 60, "minutes": 1 / 60,
        "h": 1, "hr": 1, "hrs": 1, "hour": 1, "hours": 1,
        "s": 1 / 3600, "sec": 1 / 3600,
        "second": 1 / 3600, "seconds": 1 / 3600,
    })
    data["half_life_hours_raw"] = values * factors

    # This EXACT comment describes replicate count, not a label bound.
    known_comment = (
        "pMHC-I complex stability was determined by a scintillation "
        "proximity based pMHC-I dissociation assay (PMID 21044632). "
        "The half life reported is an average of at least two "
        "independent experiments."
    )
    data["known_replicate_comment"] = data["comments"].eq(known_comment)

    # Preserve original comments. Exempt only a complete exact match;
    # additional text or other comments still receive the normal screen.
    screened_comments = data["comments"].mask(
        data["known_replicate_comment"], ""
    )
    data["comment_needs_review"] = screened_comments.str.contains(
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
        elif not np.isfinite(value):
            problems.append("nonfinite_value")
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

    # Passing the screen is NOT scientific approval.
    data["approved"] = False
    data["temperature_C"] = np.nan
    data["protocol_id"] = ""
    data["half_life_hours"] = data["half_life_hours_raw"]
    data["review_evidence"] = ""

    data.to_csv(out / "records_for_review.csv", index=False)
    group_keys = ["assay", "reference_id", "pmid"]
    counts = (
        data.groupby(group_keys, dropna=False)
        .agg(
            rows=("assay_id", "size"),
            numerical_rows=("half_life_hours_raw", "count"),
            peptides=("peptide", "nunique"),
            alleles=("allele", "nunique"),
            no_automatic_flags=("flags", lambda s: int(s.eq("").sum())),
        )
        .reset_index()
    )
    counts.to_csv(out / "counts_by_assay_and_study.csv", index=False)

    numeric = data.loc[
        np.isfinite(data["half_life_hours_raw"])
    ].copy()
    pair_columns = [
        "allele", "peptide", "modifications", "modified_residues"
    ]
    method_counts = numeric.groupby(pair_columns)["assay"].nunique()
    duplicates = data[data.duplicated("assay_id", keep=False)]
    duplicates.to_csv(out / "duplicate_assay_ids.csv", index=False)

    known_positive = (
        data["known_replicate_comment"]
        & data["flags"].eq("")
        & data["half_life_hours_raw"].gt(0)
    )
    report = {
        "input_rows": total,
        "malformed_rows_skipped": malformed,
        "human_hla_i_half_life_rows": len(data),
        "numerical_values_with_known_units": int(len(numeric)),
        "zero_values": int(data["half_life_hours_raw"].eq(0).sum()),
        "known_replicate_comment_rows": int(
            data["known_replicate_comment"].sum()
        ),
        "comments_flagged": int(data["comment_needs_review"].sum()),
        "no_automatic_flags_not_yet_verified": int(
            data["flags"].eq("").sum()
        ),
        "known_replicate_comment_positive_candidates": int(
            known_positive.sum()
        ),
        "distinct_pair_modification_groups": len(method_counts),
        "groups_with_multiple_methods": int(method_counts.gt(1).sum()),
        "duplicate_assay_id_rows": len(duplicates),
        "assay_counts": data["assay"].value_counts().to_dict(),
        "units": data["units"].value_counts(dropna=False).to_dict(),
        "temperature_verified": False,
        "checkpoint_training_overlap_audited": False,
        "sufficiency_verdict": "NOT YET ESTABLISHED",
    }
    save_json(out / "audit_summary.json", report)
    print(json.dumps(report, indent=2))
    print(f"\nReview file: {out / 'records_for_review.csv'}")
    return report
```
