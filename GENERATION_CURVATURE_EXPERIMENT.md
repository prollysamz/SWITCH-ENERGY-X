# After the measured 0.49992 result

The 25% compact-model blend scored **0.49992**, improving 0.53504 by 0.03512.
It is preserved at `candidates/submission_best_0p49992.csv`. The user's latest
reported competitor scores are 0.44323 for second and 0.27966 for first. The
user's current rank has not been verified.

The measured pair implies an optimal multiplier of approximately 1.5688 on the
previous 25% compact adjustment, equivalent to about 39.2% of that compact model.
The corresponding full-test-moment forecast is about 0.49439. This is unmeasured
and too small a gain to explain the gap to second place. The measurements do
not support replacing the baseline entirely with the compact model.

Refitting the compact families with this new exact score moves their fitted
residual variance away from zero (approximately 0.22–0.23) and their battery
coefficients close to zero. This reinforces that the earlier near-zero residual
variance was optimistic model fitting, not evidence of a recovered target law.
The diagnostic results are in `runs/compact_loss_fit_v2`. They reuse
`work/fit_compact_score_losses.py` with its output directory changed; no new
compact-law candidate was exported.

## Next structural test

The next candidate tests a negative squared-generation term: losses that grow
nonlinearly with gross generation. It uses the posterior second moment,
`E[generation²] = E[generation]² + Var(generation)`, so uncertainty is included.
This is a hypothesis about the target, not an identified transmission-loss law.

The added direction removes overlap with the incumbent, solar, demand, storage
and compact-adjustment directions on full-test moments. It has RMS 0.15 and
maximum absolute change 0.96268. No failed demand/storage adjustment is added
to the measured-best baseline.

Checks:

- On 6,000 fresh hidden generation sensors, squared-generation reconstruction
  RMSE was 8.2825 versus 114.9046 for a constant predictor (R² 0.99480).
- Separate 50,000-row projection fits gave held-out direction correlations
  above 0.99983 and relative discrepancies below 0.0194.
- All 100,000 predictions are finite, with sample-submission IDs/order and
  verified CSV round-trip values. Source files were checked against stored hashes.

These checks concern sensor estimation, numerical stability and file integrity.
They do not validate the target coefficient. The new candidate has no RMSE forecast.

Candidate: `runs/generation_curvature_v1/submission_generation_curvature.csv`.
SHA256: `1ffab46ba23c04ac466ba72126fb1f78f8a46626a11a6c405ccb608671bd816d`.

```powershell
python work/score_interaction.py EXACT_RMSE --experiment runs/generation_curvature_v1
```

Preserve 0.49992 until this experiment is measured.
