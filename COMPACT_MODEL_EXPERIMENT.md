# Compact-model experiment after the 0.58 storage result

The storage candidate returned **0.58**, recorded as rounded feedback. Neither
the failed demand nor storage adjustment is included in the new baseline.
**0.53504 remains the measured best**, with its immutable snapshot preserved.

Two approaches were audited before generating the next file:

1. Combining only exact-scored submissions gave nominal forecasts around
   0.525–0.529. Accounting for uncertain full-test/public moments made the more
   aggressive coefficients less attractive. These estimates do not establish
   private performance, and no ordinary blend was submitted.
2. A compact law with separate wind and solar terms, a demand/storage-ratio
   term, and temperature terms was fitted directly to reported RMSE values.
   Enforcing nonnegative residual variance avoided the impossible negative
   variance produced by unconstrained moment inversion. Ten exact-scored
   files entered this fit. The best compact families matched those fitting
   scores within approximately 0.006 RMSE.

The symmetric-temperature ratio model also roughly explains the recent
temperature-slope, demand and storage probes, which were excluded from fitting.
Those public diagnostics were used to select the family, so they are **not**
untouched validation. Several older approximate scores are poorly explained.
The fitted ratio coefficient is positive, contrary to a simple physical drawdown
interpretation. This is an empirical hypothesis, not a recovered physical law.

Sensitivity checks recomputed moments on forty random 30,000-row subsets while
keeping the observed scores fixed. Relative to the original compact fit, median
prediction RMS change was 0.0330, the 95th percentile was 0.3015, and the maximum
was 0.3828. The fitted variance also reached its zero boundary. These are serious
limits on trusting the full model or its implied residual RMSE; no target RMSE
forecast is reported for the candidate.

The next candidate therefore uses coordinatewise median coefficients across
those forty fits and retains **75% of the measured-best predictions**, adding
**25% of the compact-model predictions**. Its change from the best has RMS
0.13040 and maximum absolute size 1.39944. This is a limited experiment in a
coherent target formula, not another separate sensor term.

Candidate: `runs/compact_blend_v1/submission_compact_blend.csv`.
SHA256: `76b8432310b193062d874fc53df8abb83e102a7d3783a2aacee94e4885d5e602`.

The 100,000-row schema, IDs/order, finite values, CSV round-trip, blend identity,
fixed-coefficient row-order invariance and both baseline/candidate hashes were
checked. Predictions and full provenance are saved in `runs/compact_blend_v1`;
fit diagnostics are in `runs/compact_loss_fit_v1`.

```powershell
python work/score_interaction.py EXACT_RMSE --experiment runs/compact_blend_v1
```

Preserve 0.53504 until the new file is measured. Any subsequent one-direction
calibration remains adaptive public-score fitting, not hidden-label validation.
