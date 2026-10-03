# Appendix: exploratory locus transfer

Only two loci are available. Absolute correlations are low and the two-fold t intervals are highly unstable; these results do not support the headline. The per-locus bootstrap below conditions on a fixed split and resamples paired eligible alleles; it does not represent independent repeated locus experiments.

| Held-out locus | Approach | Macro ρ | Δ vs BLOSUM MLP [95% paired-allele CI] |
|---|---|---:|---:|
| A | BLOSUM MLP (reference) | 0.054 | +0.000 [+0.000, +0.000] |
| A | BLOSUM Ridge | 0.098 | +0.044 [-0.018, +0.100] |
| A | ESM-2 joint + Ridge | 0.125 | +0.071 [+0.004, +0.133] |
| A | ESM-2 joint + MLP | 0.058 | +0.003 [-0.061, +0.066] |
| A | ESM-2 mean + Ridge | 0.073 | +0.019 [-0.053, +0.086] |
| A | ESM-2 mean + MLP | 0.084 | +0.030 [-0.035, +0.093] |
| A | Allele-ID Ridge (floor) | 0.098 | +0.044 [-0.018, +0.101] |
| B | BLOSUM MLP (reference) | 0.070 | +0.000 [+0.000, +0.000] |
| B | BLOSUM Ridge | 0.043 | -0.027 [-0.081, +0.026] |
| B | ESM-2 joint + Ridge | 0.098 | +0.028 [-0.029, +0.086] |
| B | ESM-2 joint + MLP | 0.063 | -0.007 [-0.051, +0.037] |
| B | ESM-2 mean + Ridge | 0.069 | -0.002 [-0.058, +0.057] |
| B | ESM-2 mean + MLP | 0.050 | -0.020 [-0.081, +0.039] |
| B | Allele-ID Ridge (floor) | 0.044 | -0.026 [-0.080, +0.028] |

The original joint-Ridge minus BLOSUM-MLP two-locus difference was +0.050 [−0.226, +0.325]. The full seven-approach summary is in [benchmark_table.md](benchmark_table.md).

![Exploratory allele-distance and locus analysis](allele_distance.png)
