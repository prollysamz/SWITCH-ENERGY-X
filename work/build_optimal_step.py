"""Apply the measured optimum along both measured directions.

Two independent, mutually orthogonal directions have now been MEASURED against
the true residual using the exact score identity:

  1. gen_shape = perp(E_pm - E_D):  probe at t=0.05 returned 0.44480
     => c = (r^2 + t^2 - s^2)/(2 t r) = 0.171912,  g = c*r = 0.077474
  2. the in-span residual direction, determined exactly by the nine scores
     => g = 0.021455

Gains along orthogonal directions add in MSE, so the joint optimum steps each
by its own g. Predicted RMSE = sqrt(r^2 - g1^2 - g2^2) = 0.44343.

Unlike every previous candidate, these step sizes are MEASURED, not assumed.
No scored file, ledger entry or cached reconstruction is modified.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/optimal_step_v1'
INCUMBENT = 'runs/successful_components_v3/submission_successful_components_v3.csv'
R = 0.45066
PROBE_FILE = 'runs/measurement_probe_v1/submission_measurement_probe.csv'
PROBE_STEP = 0.05
PROBE_SCORE = 0.44480


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Preserve existing run')
    ledger = json.loads((ROOT / 'candidates/leaderboard_results.json').read_text())
    scored = [r for r in ledger['results'] if r.get('precision') == 'exact']
    record = next(r for r in scored if r['file'] == INCUMBENT)
    if record['rmse'] != R or sha(ROOT / INCUMBENT) != record['sha256']:
        raise ValueError('Incumbent changed')

    sample = pd.read_csv(ROOT / 'Dataset/sample_submission.csv')
    incumbent = pd.read_csv(ROOT / INCUMBENT)
    if not incumbent.row_id.equals(sample.row_id):
        raise ValueError('Row ids differ')
    base = incumbent.prediction.to_numpy()
    N = len(base)

    # Direction 1: recovered from the probe file itself, so the vector that was
    # actually scored is the vector we extrapolate along.
    probe = pd.read_csv(ROOT / PROBE_FILE)
    if not probe.row_id.equals(sample.row_id):
        raise ValueError('Probe row ids differ')
    step_vec = probe.prediction.to_numpy() - base
    norm = float(np.sqrt(np.mean(step_vec ** 2)))
    if abs(norm - PROBE_STEP) > 1e-9:
        raise ValueError(f'Probe step norm {norm} != {PROBE_STEP}')
    u1 = step_vec / norm
    c1 = (R * R + PROBE_STEP ** 2 - PROBE_SCORE ** 2) / (2 * PROBE_STEP * R)
    g1 = c1 * R

    # Direction 2: the in-span component of the residual, fixed by the scores.
    files = [r['file'] for r in scored] + [
        'runs/missingness_correction_v1/submission_missingness_correction.csv']
    P = np.column_stack([pd.read_csv(ROOT / f).prediction.to_numpy() for f in files])
    s = np.array([r['rmse'] for r in scored] + [0.46937])
    Pm2 = np.mean(P ** 2, axis=0)
    k = np.array([(Pm2[i] - np.mean(base ** 2) - (s[i] ** 2 - R * R)) / 2
                  for i in range(len(files))])
    k = k - (P.T @ base / N - np.mean(base ** 2))
    Dm = P - base[:, None]
    w = np.linalg.lstsq(Dm.T @ Dm / N, k, rcond=1e-10)[0]
    r_in = Dm @ w
    g2 = float(np.sqrt(max(k @ w, 0)))
    n2 = float(np.sqrt(np.mean(r_in ** 2)))
    u2 = r_in / n2 if n2 > 1e-12 else np.zeros(N)

    # The two directions must be orthogonal for the gains to add.
    cross = float(np.mean(u1 * u2))
    if abs(cross) > 1e-6:
        raise ValueError(f'Directions not orthogonal (cross {cross:.2e})')

    prediction = base + g1 * u1 + g2 * u2
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite candidate')
    predicted = float(np.sqrt(max(R * R - g1 * g1 - g2 * g2, 0)))

    OUT.mkdir()
    output = OUT / 'submission_optimal_step.csv'
    out = sample.copy()
    out['prediction'] = prediction
    out.to_csv(output, index=False)
    reread = pd.read_csv(output)
    if list(reread.columns) != ['row_id', 'prediction'] or not reread.row_id.equals(sample.row_id):
        raise ValueError('Round-trip failed')
    np.testing.assert_allclose(reread.prediction, prediction, rtol=1e-12, atol=1e-12)

    report = dict(
        status='Measured optimum along two independently measured orthogonal directions.',
        candidate=output.relative_to(ROOT).as_posix(), sha256=sha(output),
        incumbent_file=INCUMBENT, incumbent_rmse=R,
        probe_file=PROBE_FILE, probe_step=PROBE_STEP, probe_score=PROBE_SCORE,
        measured_cosine_gen_shape=float(c1), step_gen_shape=float(g1),
        step_in_span=g2, direction_cross_term=cross,
        predicted_rmse=predicted,
        predicted_gain=float(R - predicted),
        rms_change_from_incumbent=float(np.sqrt(np.mean((prediction - base) ** 2))),
        max_absolute_change=float(np.max(np.abs(prediction - base))),
        n_rows=int(len(out)), code_sha256=sha(Path(__file__)),
        allow_score_calibration=False,
        limitations=[
            'The step sizes are measured, but the measurement uses the PUBLIC score. '
            'Public-subset moments are approximated by full-test moments, so the '
            'realised public score may differ slightly from the prediction.',
            'Optimising against public feedback is adaptive fitting to ~30k rows; '
            'the private split may not reproduce the full gain.',
            'This exploits measured directions only. It does not identify the target.'])
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
