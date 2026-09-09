# Closed-form target search — findings

Analysis only. No submission generated, no scored file or ledger modified.

## 1. The submission set is rank-deficient

The 8 exactly-scored submissions span a **5-dimensional** subspace of R^100000.
Singular values of the 7 difference directions, normalised:

    1.000  0.2003  0.0960  0.0358  0.000  0.000  0.000

The last three submissions (0.45095, 0.45067, 0.45066) added **no new direction**.
They were re-weightings inside an already-explored subspace, which is why three
leaderboard probes bought 0.00029 of score.

Further blending of existing files cannot help. This is a closed subspace.

## 2. Leaderboard algebra gives a submission-free formula test

For any scored file P_i with public RMSE s_i, and any candidate vector c:

    mean(P_i . c) must equal (mean(P_i^2) + mean(c^2) - s_i^2)/2   for all i, if c = y

Max violation across the 8 scored files measures how far c is from the truth,
**without spending a submission**. Measured:

| candidate | max inconsistency |
|---|---:|
| pg*(1-tl)-D | 4.156 |
| pg*tau*(1-tl)-D | 3.469 |
| pg-D | 4.932 |
| E_pm | 3.327 |
| current best submission | 0.104 |

No simple closed form is close. Affine rescaling of each candidate drives the
residual down but only with unphysical intercepts (-13 to -17), i.e. the fit is
absorbing scale error, not identifying a formula.

Caveat: full-test moments substitute for unknown public-subset moments.
Difference moments are well determined (SE/mean ~1%); absolute second moments
are not (SE ~0.35 on mean(P^2)). Conclusions above rest on the difference
structure, which is the reliable part.

## 3. The binding constraint is reconstruction, not the formula

Propagating the cached posterior variances through candidate targets:

| target family | irreducible RMSE from latent uncertainty alone |
|---|---:|
| pg*(1-tl)-D | 0.1506 |
| pg*(1-tl) | 0.1473 |
| pg-D | 0.1613 |
| pg | 0.1582 |

Mean posterior sd: pg 0.0334, D 0.0241, tl 0.00083.

**The leader is at 0.10432, below every one of these floors.** So a perfect
target formula on top of the current latent reconstruction would still only
reach ~0.15. The leader necessarily recovers the latents more accurately than
this pipeline does. Formula search alone cannot close the gap.

## 4. Where the error actually is

Test rows average 6.5 of 25 sensors missing; only 0.06% are fully observed.
Mean prediction drifts monotonically with missingness (4.05 at 0 missing to
3.41 at 12 missing) — a bias signature, not noise.

MSE budget: incumbent 0.20309, leader 0.010883. Reaching first place requires
removing **94.6%** of the remaining squared error.

## 5. Implication for next steps

Priority order changes as a result of this analysis:

1. **Improve latent reconstruction on partially-observed rows.** This is the
   binding constraint. The 0.15 floor above is set by posterior variance, so
   reducing that variance is the only thing that raises the ceiling.
2. **Use the posterior mean of the target, not the target of the posterior
   mean.** For nonlinear targets these differ; the missingness-correlated bias
   in section 4 is consistent with this being uncorrected.
3. Formula refinement is worth little until (1) moves. Do not spend probes
   re-weighting the existing 5-dim subspace.

## Limitations

Public-subset moments are approximated by full-test moments. The consistency
test assumes the recorded scores correspond to the recorded files. None of this
is validated against hidden labels, which do not exist locally: train.csv has
no target column, so the leaderboard is the only label source.
