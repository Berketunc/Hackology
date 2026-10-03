# Assay reproducibility: external context, no dataset-specific ceiling

The challenge table contains one row per allele–peptide pair and no per-replicate measurements. `src/data.py` asserts pair uniqueness. Its averaged labels therefore cannot identify within-assay variance or a numerical ceiling for our macro Spearman metric. Neither SPEARMINT membership flags in `rasmussen_overlap.csv` nor conflicting measurements from the separate IEDB pilot solve this problem.

Rasmussen et al. (2016), Materials and Methods, describes the reported half-life as the **geometric mean of two independent experiments**. The paper also distinguishes the 28,166-measurement dataset from additional synthetic negatives used for its original predictor. We use the supplied table without adding those synthetic negatives. No dataset-wide replicate correlation or coefficient of variation was identified in the inspected main article. [Primary paper, methods on PDF page 3](https://discuss.iedb.org/uploads/short-url/rMbXpzwlhUyMgjK2c7yZlE44Yae.pdf).

Harndahl et al.'s assay-method paper (PMID 21044632; online 2010, issue 2011), §3.2 and Figure 3C, reports a repeat experiment on 384 HLA-A*02:01-binding peptides using fresh reagent batches, with qualitatively high agreement. The accessible text does not state a numerical repeatability correlation. Its nearby **R² > 0.95 describes the fit of individual dissociation curves**, not correlation between repeated half-life measurements. [Primary assay paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4341823/).

This provides external evidence of repeatability for that assay setting, but does not calibrate our 72-allele, zero-containing benchmark. Figure-image retrieval was unavailable during this review, so no coefficient is transcribed from the image. We do not convert qualitative agreement, curve-fit R², or an external single-allele experiment into a numerical noise ceiling. All model effect sizes remain relative to an unestimated ceiling.

Sources checked 4 October 2026. The 2012 Harndahl immunogenicity paper is distinct from the assay-method paper cited above.
