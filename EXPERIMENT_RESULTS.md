# Measured experiment results

Best confirmed public RMSE: **0.60058** (`candidates/submission_best_0p60058.csv`).

Private performance remains unmeasured. The reported leader score is approximately 0.27.

| Experiment | Public RMSE | Interpretation |
|---|---:|---|
| Preserved blend v2 | 0.71269 | Starting incumbent |
| Additional generation × temperature loss | 0.94148 | Stronger penalty rejected; small opposite adjustment supported |
| Quadratic heat above 25°C | 0.62717 | Confirmed improvement |
| Joint thermal calibration | 0.60061 | Confirmed improvement; forecast was 0.59904 |
| Observed temperature deviations | 0.64514 | No useful improvement; retain reconstructed temperatures |
| Temperature slope | ~0.84 | Rejected; rounded feedback |
| Deterministic reconstruction | 0.60058 | Confirmed but negligible gain; target weights frozen |

The temperature-slope experiment scored approximately 0.84 and was rejected.
There are no outstanding submission requests. See `DETERMINISTIC_RECONSTRUCTION.md`
for the latest completed work and rejected local hypotheses.

Temperature coefficients are being calibrated from public RMSE feedback. This is not an independent target validation set. Reconstruction validation remains based on separate held-out sensor measurements. Original scored CSV files are preserved and hash checked.
