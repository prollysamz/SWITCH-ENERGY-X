# Independent solar contribution

## Measured result

The user reported **0.53504** for this candidate. It is preserved at
`candidates/submission_best_0p53504.csv`, improving 0.60042 by 0.06538.
The two measured endpoints and full-test direction energy imply an optimal
solar-step multiplier of approximately 1.428 and a forecast RMSE of 0.52815.
This is an unmeasured public-feedback estimate, not private validation. A
separate calibration-only submission was not generated because the forecast
gain is only 0.00689. The experiment description below records the original plan.

Next candidate: `runs/direct_demand_v1/submission_direct_demand.csv`. It starts
from the measured 0.53504 file and adds a direct demand penalty with 0.20 RMS
change. The direction is orthogonal, on the full test data, to both the incumbent
and the scored solar adjustment. This keeps the measured solar adjustment fixed
for interpreting the next experiment. The target coefficient is unverified.
Demand reconstruction passed a 6,000-row hidden-sensor test (RMSE 0.06179;
constant baseline 0.16579). Independent row-partition direction correlations
exceeded 0.99992. The CSV's IDs, finite values and round-trip values were checked.
Full details and hashes are in `runs/direct_demand_v1/manifest.json`.

Confirmed best: **0.60042**, from the learned-energy submission. Its measured
gain over 0.60058 was 0.00016. The best file is preserved at
`candidates/submission_best_0p60042.csv`; learned-model weights were not recalibrated.

The next experiment changes target structure. Existing predictors derive most
generation effects from gross solar-plus-wind generation. An additional solar
coefficient has never been measured directly. About 50.6% of reconstructed solar
variance lies outside the stable directions represented by submitted predictions.
This establishes a distinct experiment, not evidence that this variance predicts NDEM.

The candidate adds 0.20 times the standardized residual solar component to the
0.60042 predictions. Overlap with substantial previous prediction directions is
removed. Base model coefficients are not refitted. Algebraically, residualizing
solar introduces compensating terms from those directions; it does not isolate
a physical intervention. The proposed positive coefficient is a hypothesis.

Checks performed:

- On 6,000 fresh observed solar readings hidden before reconstruction, sensor
  RMSE was 0.03534, versus 0.19341 for a constant prediction; R² was 0.96662.
- Projection coefficients fitted on each 50,000-row partition gave residuals
  correlating above 0.99996 with the full-data direction on the other partition.
- A projection retaining all tiny submission differences narrowly failed the
  initial stability check. Discarding directions with energy below 0.025 gave
  stable rank-eight projections and relative discrepancies below 0.009 on both
  halves. This selection used unlabeled predictions, not target scores.
- The CSV contains 100,000 finite predictions in sample-submission ID order;
  values were checked after writing and rereading it.

These checks validate component estimation, geometric stability and file
integrity. They do not identify the target coefficient or establish a new RMSE.
The actual public subset is unknown. If the added direction carried zero
residual target signal and public/full-test moments matched, the 0.20 step
would increase RMSE to about 0.63285; that is a scenario, not a forecast or bound.

Candidate: `runs/solar_contribution_v1/submission_solar_contribution.csv`.
SHA256: `ec2c74e5732c8c3d6b8f992ccccbdb76b2b242adb6841c18c5b3614604c145f3`.
Its score is unmeasured. Preserve 0.60042 until it is evaluated.

Record a returned score with:

```powershell
python work/score_interaction.py EXACT_RMSE --experiment runs/solar_contribution_v1
```

Unlike the frozen learned-model correction, this experiment may use a measured
score to estimate a single solar coefficient. Such calibration is adaptive
public feedback, not private-target validation.
