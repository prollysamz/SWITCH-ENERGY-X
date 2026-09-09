# Learned conditional models: September 9

The alternative implemented here learns corrections from masked observations.
It does not change thermal coefficients using additional public scores.
The preserved best remains `candidates/submission_best_0p60058.csv` (0.60058).

## Sensor benchmark

Each task used 60,000 training rows, 10,000 separate calibration rows and
6,000 previously unused test rows whose observed target sensor was hidden.
All target sensors and specified companion sensors were removed before inference.
Raw tree models were compared with models that learn physics residuals.
Selection and shrinkage used calibration rows only.

| Hidden sensor scenario | Physics RMSE | Selected learned RMSE | Interpretation |
|---|---:|---:|---|
| Panel temperature | 1.35007 | 1.34524 | Small, statistically unclear gain |
| Panel, loss factor and ambient temperature | 3.16011 | 3.14116 | Suggestive gain, below three paired standard errors |
| Gross generation and grid frequency | 0.59362 | 0.59802 | Worse; not evidence for replacing this reconstruction |
| All direct wind indicators | 4.61039 | 4.34230 | Clear improvement, about 11.4 paired standard errors in MSE |

Standalone tree models were worse in all four tasks. Learning physics residuals
was more effective. These results concern observed sensors withheld for testing,
not naturally missing sensor values or hidden NDEM labels.

## Whole-estimate self-supervision

The next model learns the difference between the frozen energy estimate from
richer observations and its estimate after additional observations are hidden.
Training: 60,000 rows. Calibration: 15,000 other rows. First audit: 15,000 other
rows. Confirmation: 30,000 further unused rows. Every row used for the original
physics parameter fit was excluded from these four sets. The learned residual
uses sensor readings, missing values, latent means, uncertainty and the base
estimate. No new public scores or hidden target labels enter this fit.

Rare wind cases were deliberately oversampled during masking. Reported overall
results below restore the natural test proportions: wind 0.119%, thermal 2.410%,
other 97.471%. This corrects regime prevalence, but does not make artificial
masking equivalent to natural missingness within a regime.

| Audit | Baseline recovery RMSE | Learned recovery RMSE | Paired MSE gain / SE |
|---|---:|---:|---:|
| First, 15,000 rows | 0.211824 | 0.209535 | 4.61 |
| Confirmation, 30,000 rows | 0.215395 | 0.213128 | 6.83 |

These are errors against model-generated estimates, not competition RMSE.
The teacher retains the existing target formula; this method cannot discover
target effects missing from that formula. The gain is mainly in rare wind cases.
It supports a measured experiment, not a forecast of reaching the leaders.

One confirmation row (251890) failed the existing bounded optimizer after
artificial masking. It was retained using a separate extended bounded fit with
multiple starts and explicit convergence checks. No failed fits were silently
accepted and no rows were dropped. The existing scored reconstruction and
cached test posteriors were not changed.

## Candidate and verification

Candidate: `runs/learned_energy_v1/submission_learned_energy.csv`.
SHA256: `386c114c02ce2c13055afc004ee4177abff89e30ffe703ba6d1a307e14ffed20`.
It has 100,000 finite predictions with exactly the sample submission IDs/order.
RMS prediction change from 0.60058 is 0.0280314; its competition score is unknown.
Learned weights are frozen before any public evaluation. The score recorder
will preserve a measured improvement but will not tune these weights from it.

All 16 code tests passed. Submission schema, round-trip values, baseline hash,
model hash and row-order-independent inference were checked. Full machine-readable
results, row provenance and models are in `runs/learned_sensor_v1` and
`runs/learned_energy_v1`.

Record only a real returned score:

```powershell
python work/score_interaction.py EXACT_RMSE --experiment runs/learned_energy_v1
```

Scoped public searches and competition-email searches did not locate a usable
published solution or additional organizer hint. This is not evidence that none exists.
