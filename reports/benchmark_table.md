### peptide

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.612 ± 0.013 | reference | 0.753 | +0.141 |
| BLOSUM Ridge | 0.258 ± 0.022 | -0.354 [-0.385, -0.323] | 0.556 | +0.299 |
| Allele-ID Ridge (floor) | 0.258 ± 0.022 | -0.354 [-0.384, -0.323] | 0.559 | +0.301 |
| ESM-2 mean + Ridge | 0.168 ± 0.021 | -0.443 [-0.471, -0.416] | 0.523 | +0.354 |
| ESM-2 joint + Ridge | 0.215 ± 0.029 | -0.396 [-0.424, -0.369] | 0.406 | +0.190 |
| ESM-2 mean + MLP | 0.360 ± 0.017 | -0.252 [-0.279, -0.224] | 0.596 | +0.236 |
| ESM-2 joint + MLP | 0.377 ± 0.011 | -0.234 [-0.251, -0.217] | 0.586 | +0.209 |

### allele

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.392 ± 0.126 | reference | 0.473 | +0.080 |
| BLOSUM Ridge | 0.228 ± 0.073 | -0.164 [-0.240, -0.089] | 0.246 | +0.018 |
| Allele-ID Ridge (floor) | 0.228 ± 0.073 | -0.164 [-0.240, -0.089] | 0.210 | -0.018 |
| ESM-2 mean + Ridge | 0.160 ± 0.067 | -0.233 [-0.322, -0.143] | 0.292 | +0.132 |
| ESM-2 joint + Ridge | 0.190 ± 0.081 | -0.202 [-0.274, -0.131] | 0.268 | +0.078 |
| ESM-2 mean + MLP | 0.237 ± 0.112 | -0.155 [-0.264, -0.046] | 0.282 | +0.045 |
| ESM-2 joint + MLP | 0.252 ± 0.110 | -0.140 [-0.214, -0.066] | 0.324 | +0.072 |

### locus

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.062 ± 0.012 | reference | 0.126 | +0.063 |
| BLOSUM Ridge | 0.071 ± 0.039 | +0.009 [-0.443, +0.461] | 0.101 | +0.030 |
| Allele-ID Ridge (floor) | 0.071 ± 0.038 | +0.009 [-0.438, +0.456] | 0.088 | +0.017 |
| ESM-2 mean + Ridge | 0.071 ± 0.003 | +0.008 [-0.122, +0.139] | 0.142 | +0.072 |
| ESM-2 joint + Ridge | 0.112 ± 0.019 | +0.050 [-0.226, +0.325] | 0.013 | -0.099 |
| ESM-2 mean + MLP | 0.067 ± 0.024 | +0.005 [-0.317, +0.327] | 0.092 | +0.025 |
| ESM-2 joint + MLP | 0.060 ± 0.004 | -0.002 [-0.069, +0.065] | 0.009 | -0.051 |

### allele_strict

| Arm | Macro ρ ± fold SD | Δ vs MLP [95% CI] | Pooled ρ | Pooled − macro |
|---|---:|---:|---:|---:|
| BLOSUM MLP (reference) | 0.147 | reference | 0.215 | +0.069 |
| BLOSUM Ridge | 0.080 | -0.066 [-0.191, +0.066] | 0.124 | +0.044 |
| Allele-ID Ridge (floor) | 0.083 | -0.064 [-0.188, +0.068] | 0.081 | -0.002 |
| ESM-2 mean + Ridge | -0.024 | -0.170 [-0.273, -0.061] | -0.010 | +0.014 |
| ESM-2 joint + Ridge | 0.009 | -0.138 [-0.256, -0.005] | 0.056 | +0.048 |
| ESM-2 mean + MLP | -0.027 | -0.174 [-0.275, -0.088] | 0.106 | +0.134 |
| ESM-2 joint + MLP | 0.015 | -0.131 [-0.218, -0.041] | 0.083 | +0.067 |

Peptide/allele: paired t intervals over five folds. Locus summary: two-fold t interval, highly unstable; see per-locus paired-allele intervals. Strict: paired-allele bootstrap conditional on fold 0. Intervals do not account for shared training data or method selection.
