# Battery state contribution

The direct-demand candidate returned a user-reported **0.57**. This was recorded
as rounded feedback. The best remains **0.53504**, preserved unchanged at
`candidates/submission_best_0p53504.csv`.

With a full-test direction energy of 0.04, a demand endpoint in [0.565, 0.575]
implies a nominal optimal step multiplier between approximately -0.0545 and
0.0880. The corresponding forecast gain is less than 0.0003. This rounding-range
calculation supports discarding the added demand adjustment; it does not prove
the true demand coefficient is zero. No demand calibration was generated.

The next experiment tests a positive direct contribution from battery state of
charge, beyond the already represented demand/storage ratio and other effects.
Its direction is protected against changing the incumbent, measured solar
direction, or failed demand direction on full-test inner products. The proposed
step has RMS 0.20. Its sign and strength in the actual target are unverified.

Local checks:

- A 6,000-row observed-sensor holdout gave state-of-charge RMSE 0.08663 versus
  a constant baseline of 0.17305 (R² 0.74938). These rows excluded the preliminary
  6,000-row check and previously logged sensor holdouts.
- Projection fits on two separate 50,000-row partitions gave held-out direction
  correlations above 0.99988 and relative discrepancies below 0.0154.
- Candidate IDs/order match the sample submission; all 100,000 predictions are
  finite and passed the CSV round-trip check. The best file's hash was verified.

Battery reconstruction is weaker than solar reconstruction. The adjustment
also has heavier tails: its largest absolute change is 3.57, and the largest
1% of adjustments account for about 29.4% of its squared magnitude. Under a
random 30% public split, the estimated SE of the squared adjustment is 0.00097
around a full-test value of 0.04. The public split is unknown. These are experiment
diagnostics, not evidence that this target change improves RMSE.

Candidate: `runs/storage_contribution_v1/submission_storage_contribution.csv`.
SHA256: `f972b2d304af43d984f144bb8d78f53cbc997f02e4867bb3e980c6a288489a3c`.
No new target RMSE is forecast. Preserve 0.53504 until the candidate is measured.

```powershell
python work/score_interaction.py EXACT_RMSE --experiment runs/storage_contribution_v1
```
