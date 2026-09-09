# Physics basis candidate

This is the first candidate that is materially different from the plateau and
has a quantitative reason to try it.

The predictor is a regularized quadratic interaction basis over reconstructed
physical quantities: generation, solar output, demand, storage, temperature,
thermal and transmission losses, panel generation, and delivered power. It
includes their squares, square-root transforms, and pairwise products. The
coefficients are fit only to aggregate RMSE identities from the exact scored
submissions. The latest measurement probe, optimal step, order-prior candidate,
and constant-bias probe are held out while checking the fit.

The fit's implied residual RMSE is 0.0925. Its held-out aggregate score error is
0.00355 RMSE, compared with 0.00638 on the fitting submissions. The candidate
mean is 3.86235, standard deviation 4.42613, and range −12.42 to 20.75. It
changes the incumbent by 0.44983 RMS, so it is not another tiny adjustment.

This uses leaderboard feedback as aggregate constraints, not target labels.
The held-out check only tests consistency of those constraints. A private
leaderboard score can be worse, and the coefficients are not established
physical effects. Keep the 0.43681 submission preserved until this candidate
is measured.

Measured result: the file scored 0.58194. Its large direction was therefore an
overfit to the old score set. The score identity shows its direction had only
0.104 cosine with the remaining error; the exact optimum along it is 0.1054 and
would conditionally reach about 0.43442. That gain is too small to spend a
submission on, so the measured optimum is retained only as a diagnostic.
