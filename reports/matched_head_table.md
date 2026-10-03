# Matched-head follow-up

Full-budget macro within-allele Spearman; five-fold paired t intervals. These are descriptive contrasts of fitting procedures, not a causal decomposition.

| Representation | Peptide Ridge | Peptide MLP | Allele Ridge | Allele MLP |
|---|---:|---:|---:|---:|
| BLOSUM | 0.258 | 0.612 | 0.228 | 0.392 |
| ESM-2 mean | 0.168 | 0.360 | 0.160 | 0.237 |
| ESM-2 joint | 0.215 | 0.377 | 0.190 | 0.252 |

| Contrast (first minus second) | Peptide Δ [95% CI] | Allele Δ [95% CI] |
|---|---:|---:|
| BLOSUM MLP (reference) − BLOSUM Ridge | +0.354 [+0.323, +0.385] | +0.164 [+0.089, +0.240] |
| ESM-2 mean + MLP − ESM-2 mean + Ridge | +0.192 [+0.179, +0.204] | +0.078 [+0.006, +0.150] |
| ESM-2 joint + MLP − ESM-2 joint + Ridge | +0.162 [+0.124, +0.201] | +0.062 [+0.018, +0.106] |
| ESM-2 mean + Ridge − BLOSUM Ridge | -0.090 [-0.107, -0.072] | -0.068 [-0.096, -0.041] |
| ESM-2 joint + Ridge − BLOSUM Ridge | -0.043 [-0.069, -0.016] | -0.038 [-0.050, -0.026] |
| ESM-2 mean + MLP − BLOSUM MLP (reference) | -0.252 [-0.279, -0.224] | -0.155 [-0.264, -0.046] |
| ESM-2 joint + MLP − BLOSUM MLP (reference) | -0.234 [-0.251, -0.217] | -0.140 [-0.214, -0.066] |
| BLOSUM Ridge − Allele-ID Ridge (floor) | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.000] |
| ESM-2 joint + Ridge − ESM-2 mean + Ridge | +0.047 [+0.027, +0.067] | +0.031 [-0.002, +0.063] |
| ESM-2 joint + MLP − ESM-2 mean + MLP | +0.018 [-0.009, +0.044] | +0.015 [-0.046, +0.075] |
