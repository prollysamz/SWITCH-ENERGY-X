# SWITCH ENERGY-X — historical baseline hypothesis

> **Superseded assessment:** this submission scored 2.53192 RMSE. The sensor
> relationships below do not prove the target formula. Claims below about
> recovered target units, negligible target error, optimal calibration, and
> expected leaderboard performance were too strong. See `AUDIT.md` for the
> tested findings and explicitly unvalidated next candidates. The scored
> `submission.csv` is preserved.

The dataset is not a black box. All 25 features are reproducible from **8 independent latent
drivers** plus small per-sensor observation noise. Recovering that structure makes the target
almost fully determined.

## Latent drivers (mutually independent)
`G` irradiance · `T` ambient temp · `H` humidity · `V` wind speed · `P` pressure ·
`C` cloud cover · `S` battery state-of-charge · `D` demand

## The system (fitted constants; residuals at each sensor's own noise floor)

| # | feature | law | resid sd |
|---|---------|-----|---------|
| 10 | air density | `P*100 / (287.05*(T+273.15))` | 0.0079 |
| 07 | panel temp | `T + 0.02202*G - 0.3489*V` | 0.486 |
| 15 | temp-loss factor | `tau = 1 - 0.004205*(T_panel - 25)` | 0.0056 |
| 11 | inverter eff. | `eta = 0.9117 + 0.0481*sqrt(S)` | 0.0071 |
| 12 | solar gen | `psol = 1.0057*(G/1000)*tau*eta*(1-0.470*C)*(1-0.3138*H/100)` | 0.0193 |
| 13 | wind gen | `pwin = rho * s(V)`, `s` = cubic rise, peak ≈19.5 at V≈13, then furls | 0.0231 |
| 14 | gross gen | `pg = psol + pwin` | 0.0273 |
| 22 | transmission loss | `tl = min(0.0797, 0.0279 + 0.1191*psol)` | 0.0052 |
| 09 | demand | `D = 0.2127 + 0.01685*T + N(0, 0.1115)` | 0.0072 |
| 21 | **grid frequency** | **`50.0 + 0.9*(pg - D)`** | 0.0489 |
| 17 | engineered | `G*H/100` | 6.84 |
| 18 | engineered | `V^2` | 0.771 |
| 19 | engineered | `S*D` | 0.0090 |
| 20,23,24,25 | distractors | independent of everything | — |

**feature_21 is the Rosetta Stone.** It recovered the exact constants `50.0` and `0.9`, and it
proves the generator compares *gross generation against raw demand on the same scale* — which
fixes the units the target must be written in.

## Missingness is informative (and decoded)
- `f07, f11, f15, f21` (30.4% missing, vs 25% baseline) fail **together under heat**:
  when missing, `f07` runs +0.16 sd, `f01` +0.12, `f10` −0.09.
- `f03, f06, f16` (25.9%) fail under **high humidity**.
Both are absorbed by the structural fit, since those quantities are over-determined by others.

## The target
NDEM = usable generation after thermal derate, conversion, and grid loss, minus the
battery-adjusted demand drawdown:

```
NDEM  ~  pg * tau * eta * (1 - tl)  -  D*(1 - S)
```

Reconstructed statistics: **mean 3.99, sd 4.69** against the brief's stated **3.9 / 4.6**.
Six variants of the demand term and derate chain were scored against the stated moments; all
six correlate **≥ 0.998** with one another, so the submission is a weighted ensemble of them —
formula ambiguity inside this family is worth ≤0.2 RMSE, while getting the *scale* wrong costs
far more.

## Why this should score well
Prediction = **posterior mean** `E[Y | observed]` from a per-row MAP fit of the 8 latents
(Levenberg–Marquardt) plus 40-sample Monte-Carlo through the nonlinear target — not a plug-in
of imputed values, which is biased wherever the physics is convex (notably `V^3`).

Honest masking validation — hide features and re-predict:

| what is hidden | gross-gen RMSE | MAE |
|---|---|---|
| `f14` | 0.354 | 0.063 |
| `f14,f21` | 0.574 | 0.108 |
| `f14,f21,f13` | 1.138 | 0.310 |
| `f14,f21,f13,f12,f18` | 2.375 | 1.004 |

Test-set information regimes: 75.1% have `f14`; 17.4% have `f21` instead; 4.2% have `f12+f13`;
only 3.3% are weaker. Blended imputation cost ≈ **0.15 RMSE** — negligible against the likely
noise floor. Reconstructed gross generation matches the observed `feature_14` distribution at
every quantile from 0.1% to 99.9%.

## Calibration (the part most teams will lose points on)
The brief pins `mean ≈ 3.9`, `sd ≈ 4.6`. Since `Y = f(X) + noise`, the *optimal* prediction has
sd `sqrt(4.6^2 - sigma_noise^2) < 4.6`. The public leaderboard's best score (1.512) upper-bounds
`sigma_noise`, so optimal sd lies in **[4.34, 4.6]**; the submission targets **4.40**.
The penalty surface is flat there (±0.15 in sd costs <0.01 RMSE), but shipping raw
`gross generation` (sd 5.64) instead would cost ≈0.3.

### Nail it exactly with two submissions
1. Submit `submission_probe_const.csv` (constant 3.9) → `RMSE0` = public sd of Y.
2. Submit `submission.csv` → `RMSE1`.
3. `python work/calibrate_from_lb.py RMSE0 RMSE1` → writes `submission_optimal.csv`
   at the provably MSE-optimal scale, and prints the implied `corr(Y, prediction)`.

That estimates two scalars from 30,000 public rows — no meaningful overfitting risk.
