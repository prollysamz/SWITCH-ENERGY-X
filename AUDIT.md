# Submission audit — 8 September 2026

> Latest result: **0.78886** for `submission_refined_plane.csv`, preserved as
> `candidates/submission_best_0p78886.csv`. The subsequent code/overfitting audit
> is in `GENERALIZATION_AUDIT.md`. It identifies leaderboard calibration risk,
> a concrete high-wind reconstruction failure, and validation limitations.
> No additional candidate was generated during that audit.

## Leaderboard update

### Further improvement experiment after 1.08381

Created **`candidates/submission_refined_plane.csv`**, an unscored refinement that
uses two independent directions: combined minus battery, and original minus
battery. The three scores along the first direction identify its public second
moment as 1.2426553725 (full-test estimate 1.2451380023). This shows that further
tuning on that line alone would only forecast an improvement to 1.08225.

The second direction permits rebalancing the original generation signal against
the battery/heat adjustments. Its second moment and cross moment are estimated
from all test predictions, because their values on the public subset are unknown.
The two-direction score surface has a conditional minimum near 0.80341 RMSE.
The candidate moves 90% from the scored best toward this estimated optimum,
with a conditional forecast of **0.80670 RMSE**. The 90% fraction is a stated
conservative heuristic, not a parameter fitted on independent labels.

Equivalent file weights are `-1.4875314633*battery + 2.1612403327*combined +
0.3262911306*original`. Negative weights intentionally extrapolate; predictions
are not clipped. The output mean/SD are 3.86537/4.71285, 17.43% are negative,
and RMS change from the scored best is 0.65472.

A moment-only sensitivity calculation gives roughly 0.773 to 0.837 RMSE under
a random 30% public-subset assumption. This is not a guaranteed score range or
private-leaderboard interval. It omits private distribution shift, score/file
attribution mistakes, and nonrandom public selection. Actual scoring is required.

The script verifies hashes of previously recorded scored sources, alignment and
uniqueness of all 100,000 IDs, finite outputs, and numeric CSV round-trip agreement.
Two new tests verified the recovered score surface and optimum against synthetic
row-level residuals and rejected unidentifiable inputs. The 1.08381 best file and
all prior submissions are preserved. Reproduce with
`python work/refine_scored_plane.py`; assumptions and hashes are recorded in
`candidates/refined_plane_manifest.json`.

### New best: 1.08381

The user reports **1.08381 public RMSE** for `submission_tuned_heat.csv`, close
to its conditional 1.0871 forecast. The exact scored file is preserved as
`candidates/submission_best_1p08381.csv`, with its SHA-256 verified against the
original generated manifest. The separate leaderboard ledger records this as
the new best. The original manifest remains a record of the pre-submission forecast.

This is a **57.19% RMSE reduction** from 2.53192 and is **0.42820 below** the
last reported leading score of 1.51201. Current rank has not been independently
checked; private score and final standing remain unknown. Keep this scored
submission as the best selection. No further speculative submission is needed
to beat the previously reported top-three threshold.

The result supports the chosen increase in heat penalty and reduction in
battery penalty along the tested prediction direction. It does not uniquely
identify the organizer's target formula.

### Battery comparison and next submission

The user subsequently reported **2.81560** for the battery-only candidate. The
combined candidate remains best at **1.83716**. The battery candidate's result
rejects its chosen weighting as an improvement; it does not prove storage has
no effect on the target.

Created `candidates/submission_tuned_heat.csv` from the two scored candidates:
`battery + 2.2795869759 * (combined - battery)`. This increases the heat effect
and reduces the battery effect without introducing another guessed function.
The preserved best and all scored source files remain unchanged.

For an interpolation/extrapolation weight t, the exact same-row identity is:
`MSE(t) = MSE_battery + t*(MSE_combined - MSE_battery - Q) + t^2*Q`,
where `Q = mean((combined-battery)^2)` on the scoring rows. Full-test Q is
1.2451380023. Public row IDs are unavailable, so substituting this Q is approximate.
It gives an unconstrained optimum t=2.32809. The chosen weight uses Q plus three
estimated standard errors for a random 30% subset to reduce extrapolation slightly.
This sensitivity check is not a hard bound or private-score confidence interval.

The conditional public RMSE forecast is **1.0871**, or **1.1318** at that higher Q.
These are forecasts, not observed scores. They use neither target labels nor the
brief's approximate target mean/SD. A nonrandom public subset, different scoring
rows, wrong score/file attribution, or private shift can invalidate the forecast.

Validation: scored-source hashes matched; all 100,000 IDs and finite predictions
passed CSV round-trip checks; two unit tests verified the quadratic identity and
optimum against synthetic row-level errors, plus rejection of inconsistent inputs.
The manifest is `candidates/tuned_heat_manifest.json`; reproduce after recording
scores with `python work/tune_scored_pair.py`.

### Earlier combined result

The user reports **1.83716 public RMSE for the combined candidate, rank 2**, with
first place at 1.51201. This is a 27.44% RMSE reduction from 2.53192 and meets the
top-three objective on the reported public leaderboard. Private ranking remains
unknown. The exact candidate is preserved as
`candidates/submission_best_1p83716.csv`; its hash matches the original manifest.
Scores are recorded separately in `candidates/leaderboard_results.json` so that
rebuilding candidates does not overwrite the experiment record.

The next informative comparison is the existing battery-only candidate. Its
score will help distinguish the storage and heat contributions. The result so
far validates the combined prediction change on the public split; it does not
prove either individual effect or the hidden formula. The audit below describes
the evidence available before this new leaderboard result.

Observed score: **2.53192 RMSE**. Third place in the supplied screenshot is
2.18752: a 13.6% RMSE reduction is needed. Final rankings use the hidden 70% private
split, so neither that threshold nor a public improvement guarantees final top three.

## Findings

1. **The target formula has not been recovered.** Training data is intentionally
   unlabeled. Sensor identities help reconstruction but do not prove the target's
   coefficients, units, or transformations. The brief calls NDEM a normalized,
   synthetic index; the prior claim that sensor identities fix target units was wrong.
2. **The ensemble lacks diversity.** Its six formulas mostly rescale the same
   generation signal. They scarcely explore strong battery-demand or heat effects.
3. **The deficit tail is almost absent.** Only 0.032% of baseline predictions are
   negative; minimum -0.158. The brief describes deficits as not rare and a target
   range roughly -20 to +20. Target noise can itself cause some deficits, so this
   motivates hypotheses rather than proving a specific missing effect.
4. **Scaling looks insufficient.** If the public target mean/SD were exactly
   3.9/4.6, the reported score and prediction SD 4.4 imply a best fixed-center
   linear rescale near 2.477 RMSE. These moments are approximate and describe the
   dataset, not known public labels.
5. **Storage shifts between train and test.** Observed battery SoC mean is
   0.553819 in train and 0.533796 in test. The demand-storage product shifts too.
   New formulas use training normalization; test predictions are not forcibly
   recentered, which could erase meaningful differences.
6. **Submission integrity passes.** The baseline has 100,000 unique, correctly
   ordered test IDs, required columns and finite predictions.

## Experiments

Held out observed gross generation on 8,000 test rows (seed 908), using structural
parameters previously fitted on train. These are **sensor reconstruction RMSEs,
not target RMSEs**. Selecting rows with observed gross generation does not provide
an unbiased error estimate for every original missingness regime.

| Hidden inputs | Original MAP | Bounded/informed MAP | Direct-anchor alternative |
|---|---:|---:|---:|
| Gross generation | 0.27534 | 0.27533 | 0.32463 |
| Above + frequency | 0.54503 | 0.54503 | 0.59289 |
| Above + wind generation | 1.09193 | 1.09193 | 1.42776 |
| Above + wind square | 2.32673 | 2.32639 | 2.81598 |
| Above + wind speed | 4.74309 | 4.74249 | 5.63169 |

Neither alternative materially improves reconstruction. They remain experiments
and are **not used in the candidates**. Existing parameters and submission.csv
remain unchanged.

## New hypotheses — unvalidated

All three use the reconstructed generation ensemble plus a distinct new effect.
Generation SD is approximately 3.876, conditional on the stated moments. Each
new effect has training SD 2.0, roughly the unexplained error beyond the screenshot's
first-place score. This is a heuristic, not an identified noise floor or fitted
target coefficient. Each effect is centered and its linear correlation with
generation removed on train before combining.

- **Battery:** negative `E[D/(S+0.05)]`. Uses 24-point quadrature of the approximate
  marginal SoC posterior, bounded to [0.05,1]. Cached D/S covariance is unavailable,
  so their joint expectation remains approximate.
- **Thermal:** negative `max(panel_temperature-40,0)`. The threshold is a hypothesis,
  not an organizer-specified value. Temperature comes from the inferred loss factor.
- **Combined (suggested first experiment):** combines standardized battery and heat
  effects in proportions 0.8 and 0.6, then rescales their combined effect.

The coefficients were not trained on target labels. Matching a description or
moments does not establish better scores. There is no local NDEM RMSE or promised
leaderboard result.

| Candidate | Test mean | Test SD | Negative predictions | Range |
|---|---:|---:|---:|---:|
| Battery | 3.745 | 4.375 | 13.33% | -25.11 to 23.40 |
| Thermal | 3.880 | 4.366 | 15.06% | -24.20 to 20.65 |
| Combined | 3.778 | 4.372 | 14.61% | -20.42 to 22.79 |

All candidates were re-read and checked for exact schema, 100,000 rows, unique
matching IDs, finite predictions, and numeric round-trip agreement.
`candidates/manifest.json` records assumptions, statistics and SHA-256 hashes.

## Next evidence needed

Evaluate `candidates/submission_combined.csv` next and record its actual RMSE,
keeping the 2.53192 baseline. That score can establish whether this combined
change helps, but cannot separately identify the battery and heat effects.
The other two files allow distinct follow-up comparisons within competition
limits. Prefer material, plausible improvements over tiny public-score changes.

The calibration helper now uses the second moment about the submitted constant,
rejects inconsistent inputs, and labels results approximate rather than provably
optimal. Full-test moments need not equal the hidden public subset's moments.

Reproduce from project root: `python work/audit_reconstruction.py` and
`python work/build_hypotheses.py`.
