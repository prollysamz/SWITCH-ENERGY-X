# Deterministic reconstruction and subsequent checks

The user measured **0.60058 public RMSE** for
`runs/deterministic_v1/submission_deterministic.csv`. The file is preserved as
`candidates/submission_best_0p60058.csv`, SHA256
`ef2cba635975b56601a4acfe4d4a5814afa6e7f4cb8941cf9ef6170bee530f36`.
The previous 0.60061 file also remains unchanged. The improvement is only 0.00003.

## What changed

The entire 0.60061 predictor was recovered as an explicit frozen function of
cached physical components. It reproduces the scored file to maximum error
3.18e-12, including on a partition unused for recovering the algebraic
coefficients. No hidden target labels were used for this reconstruction.

The original 40 independent Monte Carlo draws were replaced by 256 antithetic
Sobol nodes. Target coefficients, normalization, latent models and clipping were
held fixed. Results no longer depend on batching or row order.

Against a separate 4096-node numerical reference on 1,200 random test rows,
prediction RMS numerical discrepancy fell from 0.03463 to 0.000193. The high
uncertainty stress sample improved from 0.21638 to 0.00223. The reference is a
numerical approximation, not hidden-target ground truth.

Three fresh observed-sensor holdouts, each containing 6,000 rows, improved:

| Hidden sensor | MC40 sensor RMSE | Deterministic sensor RMSE |
|---|---:|---:|
| Panel temperature | 1.37638 | 1.35664 |
| Gross generation, with frequency also hidden | 0.71450 | 0.69904 |
| Solar generation | 0.035315 | 0.035117 |

All 14 tests passed. The full 100,000-row run matched the independently batched
validation predictions. All 248 fallback rows converged; none remained flagged.

The subsequent public score of 0.60058 confirms that this numerical weakness did
not explain the main leaderboard gap. No further blend was generated for the
tiny same-direction adjustment implied by the two measured scores.

## Additional hypotheses checked without submissions

Sensor failures depend on panel temperature for features 7, 11, 15 and 21, and
on humidity for features 6 and 16. Failure models fitted on train rows improved
held-out missingness likelihood, but incorporating that likelihood into
reconstruction did not show clear sensor-recovery gains in four further
holdout scenarios. It failed the predeclared improvement gate. No submission
was generated from it. See `runs/missingness_v1/validation.json`.

Small explicit energy formulas were also checked against aggregate moments from
historical public scores. The temperature-slope score was withheld from fitting.
The initial unconstrained fits implied impossible negative residual variance.
A subsequent profile across three net-generation definitions and five simple
thermal powers found no configuration passing both historical-score and held-back
score consistency checks. No formula candidate was submitted or promoted.
See `runs/formula_profile_v1/report.json`.

These are checks of limited model families, not a proof that all better formulas
are impossible. The aggregate scores remain adaptive feedback and do not provide
independent target validation. Approximate scores and unknown public row IDs
limit the precision of these calculations.

The diagnostic propagated uncertainty of roughly 0.25 under the current latent
model is **not** an identified irreducible noise floor or proof that 0.27 is
reachable. It is conditional on a possibly incomplete model and local linearity.

## Outstanding objective

A score below the reported leader's approximately 0.27 remains unachieved. The
remaining target structure is not established by the available observations and
scored formulas. Additional organizer hints or diagnostics have been requested;
further speculative submission probes have not been issued.
