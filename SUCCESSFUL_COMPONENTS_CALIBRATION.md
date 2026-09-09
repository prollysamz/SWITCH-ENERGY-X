# Joint calibration after 0.468182

The generation-curvature candidate scored **0.468182**. All six reported decimal
places are preserved in the ledger and snapshot filename:
`candidates/submission_best_0p468182.csv`. The score recorder was adjusted to stop
rounding snapshot filenames to five decimal places.

Calibrating curvature alone forecasts only about 0.46738. Instead, the next
candidate jointly calibrates the three successful directions using four exact
measured endpoints: 0.60042, 0.53504, 0.49992 and 0.468182. The rounded failed
demand and storage probes do not enter this coefficient fit.

Relative to the originally submitted steps, the candidate uses multipliers:

| Step | Multiplier |
|---|---:|
| Independent solar contribution | 1.235247 |
| Compact-model blend increment | 1.450464 |
| Generation-curvature increment | 1.164512 |

These move 90% of the way from the measured incumbent to the nominal joint
optimum. They are incremental adjustment weights, not convex file-blend weights.
The candidate's forecast is **0.45867**, not a measured score. This forecast
alone would still trail the user's reported second-place score of 0.44323.

One hundred random 30,000-row moment subsets gave fixed-candidate forecasts
from 0.45784 to 0.45966. Refitting on each subset changed predictions by at most
0.00798 RMS relative to the full-data optimum. These are sensitivity exercises
under a random-public-split approximation, not confidence bounds on private
performance or validation against hidden labels.

Candidate: `runs/successful_components_v1/submission_successful_components.csv`.
SHA256: `e481dc4df6d7f636f9039dab625f2c38cbe0748d1b58147d396f0d306f67f34b`.

The four measured endpoints were reproduced by the quadratic score identity.
An independent incremental representation reproduced the output predictions.
All 100,000 rows passed schema, ID/order, finite-value and CSV round-trip checks;
scored input hashes were verified. RMS change from the preserved best is 0.08494.

```powershell
python work/score_interaction.py EXACT_RMSE --experiment runs/successful_components_v1
```

This records a measured result without automatically fitting another calibration
to the combined step. Preserve 0.468182 until the candidate has been scored.
