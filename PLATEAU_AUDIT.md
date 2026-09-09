# Plateau audit after 0.45066

The latest measured sequence is:

| Candidate | Public RMSE |
|---|---:|
| `successful_components_v1` | 0.45095 |
| `successful_components_v2` | 0.45067 |
| `successful_components_v3` | 0.45066 |

The v2 and v3 files extend the same measured direction. Their combined gain is
0.00029, so this direction is effectively exhausted for submission purposes.

## What was checked

The exact-score blend audit now contains 23 scored files. With a conservative
eigenvalue floor, the best blend forecast is about 0.44615 versus the measured
0.45066. That improvement is inside the uncertainty of replacing the unknown
public-subset moments with full-test moments. It is not a reliable reason to
submit another blend.

A reciprocal battery-drawdown term initially appeared to forecast a much lower
RMSE. That calculation was invalid because it used the known base-prediction
cross-moment for the part of the new term that is orthogonal to every scored
submission. After projecting only the identifiable part onto the scored
prediction span, the same direction forecasts approximately 0.45063. It is
therefore not a new validated signal.

The direct physical law

```
(solar + wind) * inverter_efficiency * temperature_loss * (1 - transmission_loss)
    - demand * (1 - state_of_charge)
```

reconstructs well on complete rows, but it differs from the current public-best
prediction by about 2.6 RMS on complete test rows. Replacing the public-best
prediction with that law would be a speculative model reset, not a controlled
improvement. No CSV was produced from it.

## Decision

`candidates/submission_best_0p45066.csv` remains the submission to preserve.
Further submissions should wait until a genuinely independent target structure
is identified, such as a validated missingness-aware law or evidence that one of
the documented distractor channels leaks target information. Tiny extensions of
the successful-components direction are no longer justified.

An unlabeled test-distribution parameter adaptation was also audited separately.
It improved gross-generation sensor holdouts by only 0.0003 to 0.0015 RMSE and
slightly worsened wind and panel holdouts. It was rejected and did not alter any
scored prediction.
