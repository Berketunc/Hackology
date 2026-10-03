# Website workspace implementation

Implemented the supplied Desktop outline as the default site at `http://127.0.0.1:7860`. The original Gradio demo is preserved at `/legacy`.

The six views are connected to actual saved models: protein scan, single pair, batch upload, shortlist model comparison, saved runs and methods. The Industry design system's steel-blue palette, condensed headings, square panels and registration marks are retained, with responsive desktop/mobile layouts. The outline's simulated scoring and fabricated run history were removed.

Protein scans score all 9-mer windows and selected alleles using one of the three sequence baselines. Positions and repeated windows are preserved. Selected-pair comparisons can run all five models; new ESM-2 inputs are explicitly slower. Batch upload accepts CSV or pasted rows and validates every row before inference. Shortlists retain both peptide and allele, and CSV exports include model, central prediction, range, tier and prediction source. Browser-local saved runs restore actual inputs, results and shortlists.

Existing peptides use the appropriate peptide-held-out model, including novel allele pairings. Completely new peptides average five log-scale model predictions. Five-fit ranges remain uncalibrated diagnostics; the previously reverted conformal feature was not reintroduced. Dataset counts, zeros and exposure information come from the data rather than the design prototype.

Validation: 17 Python tests pass. Isolated Chromium checks pass for default scan (62 windows × 6 alleles), filtering, heatmap selection, shortlisting, CSV export, saved-run persistence/restoration, five-model prediction, invalid input feedback, CSV upload, batch scoring, shortlist comparison and 390px mobile layout. No JavaScript page errors or horizontal page overflow were observed. Desktop, single-pair and mobile screenshots were inspected. New-sequence ESM-2 downloading was not repeated in this UI verification; cached five-model examples and novel sequence-baseline scans were exercised.

Artifacts: `results/benchmark/workspace_checks.json`, `reports/workspace_desktop.png`, `reports/workspace_single.png`, and `reports/workspace_mobile.png`. The in-app browser connection was unavailable, so browser verification used an isolated headless Chromium installation without touching the user's browser profile.
