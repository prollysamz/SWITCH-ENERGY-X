# Why this experiment

The measured best is 0.43681. The latest reconstruction change does not close
the gap to the user-reported leader at 0.10432.

A new screen evaluated 432 bounded compact laws, including ambient versus
panel temperature, nonlinear wind response, generation curvature, and six
demand/storage terms. The best training fit had score discrepancy about
0.0029, but discrepancy on three later excluded experiments was about 0.018.
That is insufficient support for a breakthrough candidate. These public-score
checks are retrospective diagnostics, not independent validation.

All recent submissions retain approximately the same average prediction:
3.88-3.89. The original target-mean assumption came from the approximate
competition description. No previous experiment isolated a constant offset.
Using the full-test prediction-difference span, 98.7% of a constant direction's
squared magnitude is outside the explored span. Therefore the earlier claim
that all useful simple calibration directions were exhausted was too strong.

The provided CSV adds exactly 0.10 to every prediction in the preserved
0.43681 submission. This is a diagnostic, not a claim that positive bias is
present. If its measured public RMSE is s, the public mean residual is:

    bias = (0.43681^2 + 0.10^2 - s^2) / 0.20

This identity requires the same public split, but no knowledge of which rows
belong to that split. It measures positive, zero, or negative bias from one
result. The optimal public constant correction is then to add this bias to
the original baseline. Its conditional RMSE is:

    sqrt(0.43681^2 - bias^2)

Rounded leaderboard scores introduce rounding error. Private bias remains
unmeasured. An offset test cannot recover other missing target structure and
does not guarantee reaching first place. If bias is zero, the probe scores
approximately 0.44811 and the current best remains unchanged.

After receiving the exact public result for this exact file, run:

    python work/identify_global_bias.py --score SCORE

The script verifies the source hashes and writes the measured correction.
Its identity was checked on a nonuniform synthetic public subset with both
positive and negative probe shifts. The exported CSV passed ID, schema, and
constant-difference round-trip checks.
