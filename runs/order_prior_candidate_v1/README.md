# Local-prior reconstruction candidate

`submission_order_priors.csv` is unscored. The measured incumbent remains
`candidates/submission_best_0p44348.csv` (public RMSE 0.44348).

## Change

The previous model gave every row the same storage prior and the same
demand-versus-temperature intercept. Neighboring observations show that both
change with row order. A 4,001-row smoother estimates those local quantities.
Its window and residual scales were selected using observed sensors, without
target scores. Every calibration and audit row was excluded from smoothing.

The candidate replaces the two Gaussian prior factors in the cached posterior.
It applies the resulting moment changes throughout the incumbent's physics
branches. Target coefficients and normalization constants stay fixed. Learned
residual and missingness corrections stay fixed. Legacy Monte Carlo deviations
are retained through differences in deterministic moments; negative variance
approximations are clamped to zero and counted in the manifest.

The recovered formula reproduces the incumbent within 3.4e-13. Fitting its
algebra on a separate row partition changes the proposed update by less than
6e-15. This is algebraic recovery of existing predictions, not target training.

## Evidence

Confirmation on 2,000 additional rows per scenario, with settings locked:

| Hidden sensor scenario | Previous sensor RMSE | New sensor RMSE |
|---|---:|---:|
| Storage | 0.08518 | 0.06489 |
| Storage and related sensors | 0.17702 | 0.11945 |
| Demand | 0.06235 | 0.05250 |
| Demand and related sensors | 0.12058 | 0.08582 |

Separate cross-sensor checks reduced sparse-temperature RMSE from 4.6276 to
3.8574. Generation changes were statistically inconclusive.

A formula-consistency audit used richer observations as pseudo-labels, then
hid sensors. On 1,000 rows with temperature, panel temperature, storage,
demand, and generation all observed before masking:

| Masked group | Previous formula error | New formula error |
|---|---:|---:|
| Thermal | 1.28381 | 1.07300 |
| Storage | 0.01972 | 0.01886 |
| Demand | 0.01241 | 0.01088 |

An earlier version without the rich-observation requirement worsened storage
consistency (0.03323 to 0.07927). That comparison also changes uncertain
temperature estimates in the teacher rows. Both reports are retained; the
rich-observation check reduces this confound but does not supply true targets.

## Limits and decision

Prediction change is 0.06556 RMS; the largest absolute change is 2.82558.
Large changes concentrate in poorly observed thermal conditions. Gaussian
prior replacement approximates the nonlinear posterior. Artificial masking
does not establish performance under condition-dependent natural missingness.

This is a distinct reconstruction experiment supported by sensor holdouts,
not another public-score blend optimization. Hidden NDEM labels are unavailable:
there is no validated RMSE forecast and no claim this reaches first place.
Measure this one candidate before altering its target weights. Keep the
0.44348 incumbent available until a better score is observed.

Reproduction: `python work/build_order_prior_candidate.py` in a fresh run
directory, using the audited caches and scripts referenced in the manifest.
Existing run directories deliberately reject overwriting.
