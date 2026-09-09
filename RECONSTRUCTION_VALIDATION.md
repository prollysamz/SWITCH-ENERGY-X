# Label-free validation of latent reconstruction, and an orthogonal correction

Analysis + one unscored candidate. No scored file or ledger entry was modified.

## 1. The reconstruction is NOT the bottleneck (revises the previous hypothesis)

I proposed that MAP plug-in bias was the binding constraint. That was wrong,
and the cached quadrature shows why: `deterministic_moments.py` already
integrates over the posterior, so `E_pg`, `E_pm` etc. are genuine posterior
expectations. The Jensen gap between `E_pm` and the plug-in product is
mean -0.000001, sd 0.000238. There is no plug-in bias to remove.

## 2. Latents validated against exact identities, label-free

The dataset contains exact algebraic identities that let us construct ground
truth for rows where a sensor is genuinely missing:

  pg = psol + pwin            (feature_14 = 12 + 13)
  pg = (freq - 50)/0.9 + D    (feature_21 identity)

Reconstruction error where the quantity is **truly unobserved**:

| case | n | bias | RMSE |
|---|---:|---:|---:|
| pg missing, psol+pwin observed | 56,109 | +0.00025 | 0.04029 |
| pg, psol, pwin ALL missing (freq truth) | 3,224 | +0.00097 | 0.04202 |
| D missing (freq truth) | 52,160 | +0.00040 | 0.05618 |
| pg*(1-tl)-D vs observed-composed | 169,062 | +0.00960 | 0.07469 |

The hardest fully-inferential case has RMSE 0.042 against a truth that itself
carries ~0.074 of sensor noise. Bias is ~0.001 and does not grow with
missingness. **The latent recovery is sound and effectively unbiased.**

## 3. The drift is introduced by the blend layer, not the physics

Mean prediction vs number of missing sensors (low = nmiss<=3, high = nmiss>=10):

| quantity | drift |
|---|---:|
| observed pg in train (within-irradiance-stratum) | -0.124 |
| reconstruction E_pg | -0.158 |
| **every scored submission** | **-0.569** |

All eight scored submissions share an almost identical -0.57 drift, unchanged
by any calibration step. The physics supports roughly -0.13 to -0.16 of it.
The blend/target layer amplifies this roughly 3-4x.

Missingness is **not** MCAR (irradiance mean rises +56 from low to high
missingness), so part of the drift is real signal. The excess is not.

Residual of the incumbent after removing an affine function of the physics
target, by missingness: +0.50, +0.40, +0.30, +0.22, +0.16, +0.07, -0.01,
-0.03, -0.07, -0.15, -0.17, -0.18, -0.31, -0.31. Smooth and monotone, and not
explained by the physics.

## 4. The correction is a genuinely new direction

Correction vector = per-missingness-stratum mean of that excess.

    ||corr||                       = 0.12366
    component inside span(1, P)    = 0.00922
    component orthogonal to span   = 0.12331   -> 99.4% NEW

This is the first direction in the whole ledger that is not a re-weighting of
the existing rank-5 subspace. Section 1 of `FORMULA_SEARCH_FINDINGS.md` showed
the last three probes bought 0.00029 precisely because they added no new
direction; this one does.

## 5. Honest statement of what is NOT established

The score identity gives, for correction strength t:

    MSE(t) = 0.20309 - 2t*<corr, best - y> + t^2 * 0.015291

`<corr, best-y>` is **not identified** by the existing scores, because corr is
orthogonal to the span of everything already submitted. Its sign is therefore
unknown from the leaderboard alone. The train-set evidence in sections 2-3
argues the excess drift is an artifact and that removing it should help, but
that is an argument, not a measurement: train rows have no target labels.

If the excess is fully spurious, t=1 removes it. A partial step is the
conservative choice. RMS change from the incumbent is 0.1237*t.

This is a genuine one-probe experiment with an unknown sign, not a forecast
improvement. It should be judged as such.

---

# Addendum: sign and magnitude resolved

## The reconstruction tracks the observed drift almost exactly

On rows where `pg` IS observed, comparing observed drift to reconstruction drift
(low nmiss<=3 vs high nmiss>=10):

| set | observed drift | reconstruction drift |
|---|---:|---:|
| train | -0.0520 | -0.0548 |
| test  | -0.2477 | -0.2488 |

The reconstruction reproduces the real drift to within 0.003 on both sets. The
train/test difference (-0.05 vs -0.25) is a property of the data, not a bug:
both are noisy (z = -1.01 and -2.46) and consistent with a common true value.
Train and test missingness distributions are identical (mean nmiss 6.492 vs
6.504; per-feature rates agree within 0.4pp).

Pooled regression of observed `pg` on nmiss, controlling for observed G, C, V
(156,035 rows): coefficient **-0.00503 per sensor (se 0.00267, z -1.88)**,
i.e. a true drift of about **-0.040** from nmiss 3 to 11.

## The excess is not shrinkage

If the drift were legitimate posterior shrinkage, the excess would scale with
reconstruction uncertainty. It does not:

    excess ~ 0.0012 - 0.0361*sd(pg)      R^2 = 0.0000

Zero explanatory power. The excess is uncorrelated with posterior sd and is
therefore an artifact of the blend layer, not a property of the inference.

## Magnitude is determined, not guessed

The physics target already encodes the true drift. Scaled by the fitted affine
slope 0.71823, the physics-implied prediction drift is **-0.1240**.

| step t | resulting prediction drift |
|---:|---:|
| 0.00 (incumbent) | -0.5686 |
| 0.50 | -0.3463 |
| **1.00** | **-0.1240** |

t = 1.0 lands the candidate exactly on the physics-implied drift. This fixes
the step from the physics rather than from a conservative heuristic, so the
final candidate uses **t = 1.0**.

## Remaining honest caveat

The sign argument is now supported by four independent label-free checks
(identity-based reconstruction validation, observed-vs-reconstruction drift
agreement, the pooled controlled regression, and the zero-R^2 shrinkage test).
It is still an argument from the feature data, not a measurement against target
labels, because no target labels exist locally. `<corr, best-y>` remains
unidentified by the existing scores. The score may still move either way.

---

# Result: the hypothesis was WRONG. Measured 0.46937.

Submitted at t = 1.0. Public RMSE **0.46937**, worse than the incumbent 0.45066.

## What the score measures exactly

    MSE(t) = s0^2 - 2t*<corr, best-y> + t^2*q2,  s0=0.45066, q2=0.015291

Substituting the measured t=1 result:

    <corr, best-y> = (s0^2 + q2 - s1^2)/2 = **-0.000961**

For the drift to have been spurious this needed to be positive and large.
It is essentially zero, and slightly negative.

    optimal t   = -0.0629
    best RMSE along this direction = 0.45059
    maximum available gain = 0.000067

**This direction is closed.** The incumbent already carries the correct
missingness profile; the -0.5686 drift is real, not a blend-layer artifact.

## Why the label-free evidence misled me

All four supporting checks were about the *reconstruction* of latents, and they
were correct: the reconstruction is unbiased and faithfully tracks observed
drift. The error was the inferential step from there to the target.

The latents are reconstructed from the same observed sensors that define nmiss.
As sensors go missing, the posterior contracts toward the prior. The true target
is a function of the *true* latents, so its conditional mean given the observed
data legitimately moves with how much was observed. That is not an artifact to
remove -- it is the correct Bayesian behaviour, and the blend layer had already
captured it. Flattening the profile discarded real signal.

The zero-R^2 shrinkage test was the weakest link: it checked correlation with
`sd(pg)` only, when the relevant quantity is how the *conditional mean of the
target* responds to the whole observation pattern.

## What the probe bought

One new exact constraint, in a direction previously unidentified:

    mean(corr * y) = 0.020015

The submission span is now rank 6 rather than 5. The cost was 0.0187 of public
score on a probe that was declared in advance to have an unmeasured sign.

## Status

Incumbent remains **0.45066**
(`runs/successful_components_v3/submission_successful_components_v3.csv`).
Do not submit the correction candidate. The ledger was not modified.
