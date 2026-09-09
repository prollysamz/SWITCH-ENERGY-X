# Code and overfitting audit after 0.78886

> Follow-up implementation: the authorized wind-range/failed-fit repair has now
> been completed and separately sensor-validated with frozen target coefficients.
> See `RECONSTRUCTION_REPAIR.md`. The audit below records the pre-repair findings;
> the original 0.78886 file remains preserved.

## Verdict

The approach is mathematically coherent, and the audited pipeline does not show
row-specific fitting to hidden target labels. **It does fit global coefficients
using public leaderboard feedback.** The public split therefore is not an
independent validation set. We cannot certify absence of overfitting or predict
the private score from the available data.

The evidence does not suggest that the large gains are merely the result of a
high-capacity model memorizing public rows. All five scored prediction files lie
in a two-dimensional affine space: recent tuning adjusts two global directions.
The corresponding moment matrix has condition number 3.96, without severe
numerical instability. This is limited tuning capacity, not a bound on all
earlier adaptive hypothesis selection or a statistical guarantee.

Further improvement is plausible. **Fix identifiable reconstruction weaknesses
and validate them on held-out sensors before more public-score tuning.** No new
candidate was generated during this audit. The 0.78886 file was preserved as
`candidates/submission_best_0p78886.csv` and recorded in the score ledger.

## Findings, in priority order

### 1. High-wind behavior is outside the reconstruction model's support

**Code:** `work/model3.py:22-34`, especially `Params.wind`; `work/model3.py:81`.

The wind spline clips speed to [0,18], but observed speed exceeds 18 on 34 train
and 14 test rows, with a test maximum of 21.76. The reconstruction can compensate
for the wrong wind prediction by distorting temperature and pressure because the
latent optimizer has no physical bounds or residual-quality rejection.

Concrete case: test **row_id 438963** has measured wind speed 21.12 and a measured
temperature-loss factor 0.910715, implying a panel temperature about 46.23 using
the fitted relation. The optimizer instead returns ambient temperature 115.49,
pressure 893.80 and panel temperature 130.18. The cached panel estimate is about
130, and the resulting best target prediction is **-29.08**. This is a clear
contradiction between the reconstruction and an available observed sensor.

The previous experimental model4 clips speed to 18 as well and gives about 95.61
for this panel temperature, so simply enabling that rejected model is not a fix.
The production model should support the observed wind domain and handle
inconsistent sensor fits; any bounds should be justified by data and physics.
This finding is a numerical/modeling defect, not proof of leaderboard overfitting.

### 2. Independent target validation is absent

**Code:** `work/refine_scored_plane.py:46-84`, `work/tune_scored_pair.py:39-66`;
validation in `work/validate.py:6-28` and `work/audit_reconstruction.py:36-54`.

The score helpers infer target cross-moments from public RMSE. Statements such as
"no labels used" should be understood only as **no individual labels accessed**:
aggregate information about hidden targets is being used to fit the predictor.
Reusing a holdout adaptively creates overfitting risk, even with mathematically
correct score calculations. This is the general problem studied by
[Blum and Hardt, The Ladder (ICML 2015)](https://proceedings.mlr.press/v37/blum15.html).

The older validate.py uses a train sample that is not explicitly disjoint from
the 120,000-row parameter-fitting sample. Later sensor audits use test rows with
an observed sensor hidden during inference, which avoids that specific overlap.
However, none of these tests evaluate the hidden NDEM target. Sensor observations
are also selected by their availability, so tests do not reproduce all naturally
missing sensor regimes.

Pre-submission forecasts of 1.08715 and 0.80670 were followed by public scores
1.08381 and 0.78886. That supports the score-surface calculation on the same
leaderboard, **not independent generalization to the private split**.

### 3. Missingness mechanisms are not explicitly modeled

**Code:** `work/model3.py:56-66`.

The model uses available observations and fixed priors. It omits missing residuals
but has no likelihood term for the probability a sensor is missing under heat or
humidity. Redundant surviving measurements can still recover the state; this is
different from learning information conveyed by a failure pattern itself. The
historical claim that missingness was fully decoded/absorbed was too strong.

There is observed storage shift: mean feature_08 is 0.55382 in train versus 0.53380
in test; the standardized shift is -0.1153. This is train-to-test evidence only,
not evidence of public-to-private shift. The largest shift is the documented
distractor feature_24, which the active model does not use.

### 4. Nonlinear uncertainty calculations are approximate

**Code:** `work/build_hypotheses.py:16-28,48-49`; `work/run_full.py:5,20-35`.

The heat term is `max(E[T]-40,0)`, not `E[max(T-40,0)]`. A diagnostic using a
Gaussian marginal posterior finds an RMS difference of 0.02768 prediction units,
with a maximum of 1.68 on individual rows. This is a possible correction's size,
not an expected RMSE improvement. The storage term also omits joint D/S covariance.
Production block estimates use only 40 Monte Carlo draws and a local Gaussian
approximation around a single optimizer solution.

The beta used to reconstruct panel temperature is hardcoded to 0.004213, whereas
the saved fitted parameter is 0.004205404957. This is a small consistency issue;
changing it without checking downstream calibration is not automatically better.

The final standardized storage penalty is only about 0.16975, versus a heat
penalty of 2.35861. Thus the large negative CSV combination weights do not imply
an enormous physical battery penalty; much of that coefficient cancels algebraically.

### 5. Forecast sensitivity excludes important sources of uncertainty

**Code:** `work/refine_scored_plane.py:73-95`.

The code substitutes full-test moments for unavailable public-subset moments and
assumes a random 30% public sample for its sensitivity calculation. This is a
moment-only sensitivity calculation, not a confidence interval for private RMSE,
nor an adjustment for repeated adaptive model selection. The 90% step toward
the estimated optimum is a heuristic; it was not chosen by independent validation.

Tail rows matter: the highest-influence 1% of rows account for about 15.24% of the
two-direction regression leverage. A small number of sensor reconstruction failures
can therefore affect calibration more than their frequency suggests.

### 6. Reproducibility relies too much on existing file order

**Code:** `work/run_full.py:41`, `work/build_hypotheses.py:85`,
`work/refine_scored_plane.py:99`.

The .npz caches contain no row IDs, input hashes, or model version metadata. They
are assumed to match CSV order. Candidate builders overwrite fixed output names;
training scripts also write parameters/caches to fixed paths. Source hashes and
immutable copies protect the currently recorded best, but do not make all future
reruns safe. Versioned caches with explicit IDs, input/model hashes and output
overwrite protection are appropriate before further changes. The checks use
Python assertions, which disappear with `python -O`; current audited runs did
not use that flag.

## Checks actually performed

- Verified hashes, exact schema, 100,000 unique aligned IDs and finite predictions
  for all five scored files. No prediction model uses row_id as a fitted feature.
- Rebuilt the original ensemble from cached components: maximum difference
  1.07e-14. Rebuilt the best file from its recorded weights: 3.55e-15.
- Ran all four existing score-math unit tests successfully. These use synthetic
  labels to test identities and invalid-input rejection, not real NDEM accuracy.
- Audited latent bounds, model domain, active variables, source provenance,
  uncertainty propagation, missingness and prediction-tail behavior.
- Ran fresh panel-temperature sensor holdouts on 5,000 test rows, seed 910:

| Hidden sensors | Panel RMSE | MAE | Largest absolute error |
|---|---:|---:|---:|
| Panel temperature | 1.3160 | 0.9355 | 9.75 |
| Panel temperature + temperature-loss factor | 2.7975 | 1.2421 | 141.13 |
| Above + ambient temperature | 3.7931 | 2.4322 | 141.13 |

These are sensor units, not target RMSE. They identify rare unstable fits and do
not prove that changing reconstruction improves NDEM. The saved cache also has
23 test rows with inferred SoC above one; the train cache has five negative
inferred demand values. A noisy proxy can exceed physical bounds, but unbounded
latent states require attention rather than automatic acceptance as posterior truth.

## Recommended next sequence

1. Freeze and retain the 0.78886 scored file, current model artifacts and weights.
2. Correct wind-domain/failed-fit handling and add regression tests for the
   contradictory tail row and high-wind sensor holdouts.
3. Validate these changes on disjoint sensor holdouts stratified by wind, heat,
   missingness and storage. Keep target coefficients fixed during that work.
4. Evaluate joint uncertainty and missingness-aware reconstruction if those tests
   justify it. Reuse fitted parameters consistently and version the caches.
5. Only then make one controlled candidate comparison. Do not promise a lower
   private RMSE or keep adding arbitrary transformations to chase tiny public gains.

There is no justified numerical forecast for those unimplemented changes.
Audit diagnostics are reproducible with `python work/audit_generalization.py`
and saved in `work/generalization_audit.json`.
