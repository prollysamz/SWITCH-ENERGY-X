# Audit after the unsuccessful temperature-slope submission

The best confirmed public score remains **0.60061**. Its immutable snapshot is
`candidates/submission_best_0p60061.csv`, SHA256
`79347b19112365760de56159341c3a2f92c17e4551d2af02c9afbe814bff6e9b`.
The latest slope score was reported as approximately 0.84 and is recorded as
rounded feedback, without fitting another candidate from it.

## Main finding: weak candidate selection

The latest files tested unverified target effects. Their coefficients were chosen
to make effects measurable, not because local evidence predicted improvement.
For example, adding an orthogonal change with standard deviation 0.5 contributes
0.25 to MSE if it has no relationship with the remaining target error. Against
0.60061, that would yield approximately 0.782 RMSE on matching scoring moments.
The public subset differs from the full test set, so this is an illustrative
calculation, not a forecast for the actual submitted file.

These submissions were information-gathering probes. Presenting successive probes
as the route to an improved submission was a poor choice for the user's request.
Their lack of support is not a row-alignment bug. Public feedback is being used
for training and model selection; there is no independent NDEM validation set.
No more speculative probe submissions were produced after this audit began.

## Checks completed

All 13 scored source CSVs passed available SHA256 checks, exact sample row order,
unique IDs, schema and finite-value checks. The best snapshot exactly matches
the file that scored 0.60061.

Four fresh sensor holdouts each used 5,000 rows, excluding all previous sensor
audit samples. Increasing optimization from 22 to 66 iterations made negligible
differences:

| Hidden sensors | 22-iteration sensor RMSE | 66-iteration sensor RMSE |
|---|---:|---:|
| Panel temperature | 1.357034 | 1.357034 |
| Panel temperature and loss factor | 1.834918 | 1.834893 |
| Above plus ambient temperature | 3.218978 | 3.218976 |
| Gross generation and grid frequency | 0.669829 | 0.669829 |

These are sensor errors, not target RMSE. More optimizer iterations do not provide
evidence for an improved submission. The 66-iteration stress run also produced
damping-overflow warnings, reinforcing that simply increasing its fixed loop
count is not an appropriate production change.

The 40-draw Monte Carlo cache adds measurable numerical noise to panel-temperature
means: RMS difference from the analytic affine Gaussian mean was **0.12723°C**
on 7,982 ordinary rows, versus **0.12792°C** expected sampling RMS. This is a
quantified numerical weakness, not evidence that it explains the remaining
0.60061 target error or that fixing it reaches 0.27.

## Concrete repair

`work/reconstruction_repair.py` previously accepted a fallback optimizer result
based on finite cost without checking `fit.success`. It now rejects unsuccessful
or nonfinite fits, reports a `failed_fallback` flag, and still refuses invalid
latent states. The original source was preserved in
`runs/pipeline_audit_v2/reconstruction_repair_before.py`.

Every fallback call in this fresh audit converged. Therefore this guard fixes a
real handling gap but does **not** explain the recent poor submission scores.
Two regression tests cover unsuccessful fits and invalid states. All 12 tests
pass. Scored predictions and model parameters were not regenerated or overwritten.

Full numerical results and held-out row IDs are saved in
`runs/pipeline_audit_v2/audit.json` and `holdout_rows.json`.

The reported leader score of approximately 0.27 has not been achieved. No claim
of private leaderboard improvement or guaranteed future public score is supported.
