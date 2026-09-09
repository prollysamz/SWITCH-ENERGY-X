# Reconstruction repair with frozen target weights

The **0.78886** submission is preserved and hash-verified. The new candidate is
`runs/reconstruction_v1/submission_reconstruction_repaired.csv`. It is **unscored**.
No leaderboard score was used to fit the repaired wind curve or select its parameters.

## What changed

- Retained the existing wind spline exactly below speed 14. Between 14 and 15,
  smoothly transition to a positive logistic tail that continues across the
  observed high-speed range instead of clipping every speed above 18 to one value.
- Fitted three tail parameters using 1,282 training rows with observed wind-power,
  air-density and wind-square sensors. A deterministic 300-row training holdout
  was used during model selection. An initial quadratic log-tail was rejected
  because it missed the highest-wind cases; the three-parameter logistic tail was
  selected for its better tail error and stable decreasing extrapolation.
- Detect fits with out-of-bounds latent states or sensor residuals above eight
  nominal noise standard deviations. Retry from both an observation-informed
  start and the existing solution using bounded robust least squares. Bound the
  posterior samples of repaired fits as well. Any unresolved inconsistencies are
  explicitly reported rather than silently treated as reliable estimates.

These checks are modeling safeguards, not calibrated probabilities of fit failure.
Bounds allow wind speed up to 30, covering the observed maximum of 21.76. The
new tail is not an assertion that the organizer used this exact physical law.

## What stayed fixed

The target formula, both original normalization schemes, generation scale,
battery/heat transformations, the hardcoded heat conversion, and final tuned
weights were frozen. In particular, final weights remain:

`battery + 2.161240332695635*(combined-battery)
         + 0.3262911305744569*(original-battery)`.

The existing 40 posterior samples, seed 2024 and 25,000-row chunking were retained
to keep random draws comparable. The old target decoder reproduced the scored
best before the reconstruction changed. No target moments were recomputed from
the new predictions, and no coefficient was retuned against the leaderboard.

## Fresh sensor holdouts

After selecting the tail on training data, evaluated on fresh test sensor rows
using seed 921. Rows from the previous 8,000-row gross-generation and 5,000-row
temperature audits were excluded, as was known regression row 438963. Rare
high-wind cases were deliberately included and are reported separately.
The all-row cohorts therefore are not prevalence-weighted population estimates.

| Hidden sensors / predicted sensor | Rows | Original RMSE | Repaired RMSE |
|---|---:|---:|---:|
| Panel temperature / panel temperature | 6,230 | 1.60270 | 1.42487 |
| Panel temperature + loss factor / panel temperature | 6,230 | 2.25193 | 1.87150 |
| Above + ambient temperature / panel temperature | 6,230 | 3.86550 | 3.20208 |
| Gross generation / gross generation | 6,257 | 0.33673 | 0.33366 |
| Gross generation + frequency / gross generation | 6,257 | 0.56986 | 0.56611 |
| Wind generation / wind generation | 6,262 | 0.36283 | 0.35917 |

For the 11 rows with wind above 18 in the panel-temperature cohort, RMSE fell
from 15.73159 to 0.87492 when panel temperature was hidden, and from 27.96536
to 1.51308 when its loss-factor proxy was hidden too. These small tail samples
are encouraging but do not provide a precise estimate of future rare-event error.

Ordinary-wind, high-heat and low-storage strata were also checked. Ordinary-wind
errors were essentially unchanged: the largest observed increase was about
0.00003 sensor units in the hardest temperature scenario. The high-heat and
low-storage strata improved in each reported scenario. No further parameter
selection was performed from these fresh test holdout results.

**All these metrics concern input sensors, not NDEM target RMSE.**

## Full test reconstruction

- Rebuilt all 100,000 rows. Retried 248 suspect fits; all 248 accepted a bounded
  replacement, and none remained above the residual flag threshold.
- Row 438963 now has inferred panel temperature 47.33, versus 130.18 originally;
  its observed loss-factor sensor implies about 46.23. The repaired posterior
  target prediction is **3.54964**, versus **-29.08229**.
- 1,388 predictions changed by more than 1e-8; **98.612% remained unchanged within
  that tolerance**. Only 76 changed by more than 0.25.
- RMS prediction change is 0.10797. This is not a forecast RMSE improvement.
- New prediction mean/SD: 3.86591 / 4.71202. Range: -11.53264 to 21.70102.

## Verification and artifact protection

Eight tests pass: the existing four score-math tests plus wind-domain/join
regression, the known failed row, explicit flagging of an inconsistent synthetic
sensor, and exact reconstruction of the scored best with frozen target settings.
The generated CSV was read back and checked for 100,000 unique matching IDs,
correct schema, finite predictions, and numerical round-trip agreement.

Legacy datasets, caches, parameters, scored best and calibration manifests were
hashed before and after the run and remained unchanged. New caches carry row IDs.
The versioned output directory refuses overwrite, and the run manifest stores
input/source hashes, frozen target settings, cache/output hashes and diagnostics.

Artifacts:

- `runs/reconstruction_v1/submission_reconstruction_repaired.csv` — candidate.
- `runs/reconstruction_v1/manifest.json` — provenance and measured changes.
- `runs/reconstruction_v1/frozen_target.json` — exact frozen target configuration.
- `runs/reconstruction_v1/affected_rows.csv` — retries and material changes.
- `work/reconstruction_repair_validation.json` — stratified holdout results.

The repaired prediction still uses approximate local Gaussian uncertainty and
does not explicitly model missingness mechanisms. Those were intentionally not
changed in this experiment. Keep 0.78886 as the best until this candidate has an
actual score; sensor improvement cannot guarantee public or private target gains.

## Independent verification (second pass)

Re-ran all 8 tests: pass. Re-hashed every scored artifact and every protected
input listed in the run manifest: all match, nothing was mutated. The repaired
CSV round-trips to 100,000 unique IDs in sample order with finite values, and its
hash matches the manifest.

### The change is essentially one row

`sum(delta^2)` is dominated by a single row: **438963 alone is 91.3%** of all
squared change, and the top six rows are 97.0%. The remaining 1,382 changed rows
contribute almost nothing.

| row_id | scored best | repaired | delta |
|---|---:|---:|---:|
| 438963 | -29.082 | 3.550 | +32.632 |
| 423969 | 13.725 | 19.170 | +5.445 |
| 461239 | 1.351 | 4.840 | +3.490 |
| 400970 | 16.471 | 13.061 | -3.410 |
| 452602 | 10.790 | 7.939 | -2.851 |

**Correction after handoff review:** the earlier claimed 0.0074 improvement cap
and 0.008 downside bound were incorrect. The cross-term
`2*mean(d*e_old)` is unknown. Perfect-correction and uncorrelated-noise assumptions
describe particular scenarios, not upper/lower bounds. The norm inequality gives
`abs(RMSE_new-RMSE_old) <= RMS(d)` only when all quantities use the same evaluation
rows. Full-test RMS(d)=0.10797 cannot silently replace public-subset RMS(d),
particularly when one row dominates. The public/private placement of row 438963
is not known. The reported repaired score near 0.75 is empirical evidence and
does not contradict the correct error identity.

### Score-space status: the plane is exhausted

All five scored vectors are affine combinations of the three basis predictions
(`submission.csv`, `combined`, `battery`) with weights summing to exactly 1.
Re-solving the optimal plane weights from all five scores gives
`[0.3637, 2.1486, -1.5123]`, forecasting **0.79409** against the achieved
**0.78886**. The residuals of that over-determined solve are ±0.009 in MSE units,
i.e. the full-test Gram matrix approximates the unknown public Gram to about
±0.006 RMSE. **The current weights are already at the optimum within that noise;
re-tuning inside this plane cannot help.**

Two consequences:

1. **`candidates/submission_thermal.csv` is worthless as a submission.** It lies in
   the scored affine span to a residual of 1e-6 (against its own SD of 4.37). It
   would buy no new direction.
2. **The repaired file is the only genuinely new direction available**, sitting
   0.1079 off the span. Once scored it becomes a fourth basis vector, and the
   same exact line search can then find the optimal blend between it and 0.78886 —
   which self-corrects if the repair over- or under-shot.

### What is *not* knowable from the current scores

Because every scored vector has weights summing to 1, the constant `mean(Y^2)`
cancels **exactly** from every equation. `sd(Y)` and therefore the total remaining
headroom are **not identifiable** from these five scores — an earlier suggestion
that an over-determined solve could recover them was wrong; the design matrix has
`(1,1,1,2)` exactly in its null space, and solving anyway returns a nonsensical
negative `mean(Y^2)`. A constant prediction adds another aggregate error
measurement but does not by itself identify the target mean, absolute
correlations or intrinsic noise floor. In particular, scoring constant 3.9
measures `E[(Y-3.9)^2]`, not necessarily `Var(Y)`. Unexplained error can contain
missing model structure as well as noise. Public prediction moments also remain
unknown without the actual scoring row IDs.
