# Review of intervening changes — 9 September 2026

User-confirmed current best: **0.71269**, from
`candidates/submission_blend_v2.csv`. It is preserved unchanged as
`candidates/submission_best_0p71269.csv`, SHA-256
`48c8d6801f7290839877dcaedcb559c0d65499d941e1aa86c7f80c619ddeae8f`.
The user reports public rank 3, with second near 0.69 and first near 0.27.
Private scores are unknown.

## What was verified

- The wind-repair source, frozen-target source, protected input data and scored
  artifacts still match the reconstruction run's hashes.
- All eight existing tests pass.
- Latest blend, thermal-tail candidate and repaired file all contain 100,000
  unique matching IDs, correct columns and finite predictions.
- The latest blend is reproducible from the new blend script and its preceding
  ledger state to maximum difference 3.77e-13. It is approximately 66.28% repaired
  and 36.82% thermal-tail, with small signed contributions from older submissions.
- The known failed-fit row 438963 is predicted at 4.99 in the latest blend.

## Limits and issues

1. The exact 0.71269 result was missing from the ledger; it is now recorded.
   Repaired (~0.75) and thermal-tail (~0.84) scores remain rounded/approximate.
   Those must not be described as exact target moment measurements. Assuming
   rounding uncertainty of +/-0.005 changes the earlier blend's predictions by
   up to about 0.0109 RMS in the checked endpoint scenarios.
2. With 0.71269 included, `resolve_blend.py` forecasts about 0.71206, an improvement
   of only 0.00063. That conditional estimate does not justify another blend
   submission. No new candidate was generated during this review.
3. The blend solver's 0.05 eigenvalue floor is a heuristic. It is basis-dependent
   and is not a formal proof of identifiable directions or a private-score bound.
   The solver also lacks per-input hash and row-order checks. This review checked
   current files independently; future changes should restore those safeguards.
4. The added repair report incorrectly asserted a 0.0074 cap on RMSE improvement
   and a 0.008 downside bound. Its cross-term is unknown, and public row moments
   differ from full-test moments. That passage has been corrected.
5. A constant probe does not identify the intrinsic noise floor. A score against
   constant c gives E[(Y-c)^2], which includes mean error; unexplained model error
   can include missing structure as well as irreducible noise.
6. `work/split_gen.py` reconstructs solar/wind components with old `model3` and
   `params_final.pkl`, reintroducing the clipped wind-domain model. Any next use
   of these component features should rebuild them with the repaired model and
   validate row alignment and provenance first.
7. New `moment_fit.json` entries rank formulas by descriptive-moment agreement,
   not measured target RMSE. Matching approximate target moments is not validation.
   Broadly submitting all generated thermal probes would increase adaptive
   leaderboard overfitting risk.

## Next useful direction

The current scored blend is effectively exhausted at the available precision.
Further work should test one physically motivated effect outside that blend,
such as distinguishing solar and wind conversion/thermal losses, after rebuilding
its components using the repaired reconstruction. Preserve current predictions
and isolate the effect. The leader's 0.27 alone does not identify which effect is
missing or establish a reachable noise floor. No new score is promised.
