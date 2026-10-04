# Post-hoc patience sensitivity

Triggered by selected epochs ≤2 in the original allele-regime ESM-2 MLP fits. Only patience changes (15 → 30); the 150-epoch cap, allele-grouped validation, features, optimizer, widths, and fold seeds stay fixed. All three MLP approaches are rerun. The main benchmark and headline tables remain at patience 15.

| Regime | Approach | Macro, patience 15 | Macro, patience 30 | Selected epoch ≤2 at patience 30 |
|---|---|---:|---:|---:|
| allele | blosum_nn | 0.392 | 0.392 | 0/5 |
| allele | esm2_mean_nn | 0.237 | 0.234 | 1/5 |
| allele | esm2_joint_nn | 0.252 | 0.256 | 3/5 |
| allele_strict | blosum_nn | 0.147 | 0.147 | 0/1 |
| allele_strict | esm2_mean_nn | -0.027 | -0.027 | 0/1 |
| allele_strict | esm2_joint_nn | 0.015 | 0.015 | 1/1 |

| Regime | Approach − BLOSUM MLP | Patience 15 Δ [95% CI] | Patience 30 Δ [95% CI] |
|---|---|---:|---:|
| allele | esm2_mean_nn | -0.155 [-0.264, -0.046] | -0.158 [-0.264, -0.052] |
| allele | esm2_joint_nn | -0.140 [-0.214, -0.066] | -0.136 [-0.218, -0.054] |
| allele_strict | esm2_mean_nn | -0.174 [-0.275, -0.088] | -0.174 [-0.275, -0.088] |
| allele_strict | esm2_joint_nn | -0.131 [-0.218, -0.041] | -0.131 [-0.218, -0.041] |

**Early selected checkpoints persist in 5/18 fits at patience 30.**

An epoch-1 checkpoint has already received a full epoch of gradient updates; it is not an untrained model. The selected epoch does not equal the number of epochs attempted. Persistent early selection means this patience increase did not resolve the pattern, not that an underlying cause has been identified. No patience-60 run or alternative validation grouping is performed.

Allele intervals pair five folds; strict intervals bootstrap paired eligible alleles within one fixed split. These nominal intervals do not account fully for overlapping training data, related alleles, multiplicity, or this post-hoc choice. Peptide folds were not rerun at patience 30, so their patience sensitivity is untested.

## Full-budget stopping epochs

| Approach | Regime | Fold | Training rows | Best epoch, patience 15 | Best epoch, patience 30 |
|---|---|---:|---:|---:|---:|
| blosum_nn | allele | 0 | 22879 | 4 | 4 |
| blosum_nn | allele | 1 | 20511 | 7 | 7 |
| blosum_nn | allele | 2 | 21409 | 16 | 16 |
| blosum_nn | allele | 3 | 21704 | 9 | 9 |
| blosum_nn | allele | 4 | 21621 | 23 | 23 |
| esm2_joint_nn | allele | 0 | 22879 | 1 | 1 |
| esm2_joint_nn | allele | 1 | 20511 | 9 | 88 |
| esm2_joint_nn | allele | 2 | 21409 | 1 | 1 |
| esm2_joint_nn | allele | 3 | 21704 | 21 | 92 |
| esm2_joint_nn | allele | 4 | 21621 | 2 | 2 |
| esm2_mean_nn | allele | 0 | 22879 | 1 | 1 |
| esm2_mean_nn | allele | 1 | 20511 | 18 | 18 |
| esm2_mean_nn | allele | 2 | 21409 | 9 | 9 |
| esm2_mean_nn | allele | 3 | 21704 | 22 | 39 |
| esm2_mean_nn | allele | 4 | 21621 | 4 | 4 |
| blosum_nn | allele_strict | 0 | 7794 | 9 | 9 |
| esm2_joint_nn | allele_strict | 0 | 7794 | 1 | 1 |
| esm2_mean_nn | allele_strict | 0 | 7794 | 7 | 7 |
| blosum_nn | locus | 0 | 13696 | 13 | not run |
| blosum_nn | locus | 1 | 13335 | 21 | not run |
| esm2_joint_nn | locus | 0 | 13696 | 45 | not run |
| esm2_joint_nn | locus | 1 | 13335 | 29 | not run |
| esm2_mean_nn | locus | 0 | 13696 | 6 | not run |
| esm2_mean_nn | locus | 1 | 13335 | 22 | not run |
| blosum_nn | peptide | 0 | 21576 | 24 | not run |
| blosum_nn | peptide | 1 | 21884 | 28 | not run |
| blosum_nn | peptide | 2 | 21504 | 24 | not run |
| blosum_nn | peptide | 3 | 21520 | 23 | not run |
| blosum_nn | peptide | 4 | 21640 | 15 | not run |
| esm2_joint_nn | peptide | 0 | 21576 | 13 | not run |
| esm2_joint_nn | peptide | 1 | 21884 | 11 | not run |
| esm2_joint_nn | peptide | 2 | 21504 | 20 | not run |
| esm2_joint_nn | peptide | 3 | 21520 | 56 | not run |
| esm2_joint_nn | peptide | 4 | 21640 | 8 | not run |
| esm2_mean_nn | peptide | 0 | 21576 | 8 | not run |
| esm2_mean_nn | peptide | 1 | 21884 | 8 | not run |
| esm2_mean_nn | peptide | 2 | 21504 | 7 | not run |
| esm2_mean_nn | peptide | 3 | 21520 | 8 | not run |
| esm2_mean_nn | peptide | 4 | 21640 | 15 | not run |

Machine-readable records: [fold scores](../results/benchmark/patience_sensitivity/per_fold.csv), [paired comparisons](../results/benchmark/patience_sensitivity/paired_comparisons.csv), [within-approach changes](../results/benchmark/patience_sensitivity/patience_shifts.csv), [epochs](../results/benchmark/patience_sensitivity/best_epochs.csv), and [acceptance checks](../results/benchmark/patience_sensitivity/checks.json).
