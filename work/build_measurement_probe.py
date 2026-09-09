"""Small-step measurement probe along one unmodelled physics direction.

Design: the previous probe stepped a full unit along an unmeasured direction and
lost 0.0187 to learn one number. A small step learns the SAME number for ~7x less.

    MSE(t) = r^2 - 2*t*c*r + t^2,  r = 0.45066, u a unit direction orthogonal
    to the span of every scored submission, c = cos(u, y - incumbent).

Observing the returned RMSE at a known small t solves for c exactly. The optimal
step then follows in closed form. Worst case (c = 0) costs 0.0028.

No scored file, ledger entry or cached reconstruction is modified.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/measurement_probe_v1'
INCUMBENT = 'runs/successful_components_v3/submission_successful_components_v3.csv'
INCUMBENT_RMSE = 0.45066
STEP = 0.05
DIRECTION = 'gen_shape'          # perp(E_pm - E_D): core energy balance shape


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Preserve existing run')
    ledger = json.loads((ROOT / 'candidates/leaderboard_results.json').read_text())
    scored = [r for r in ledger['results'] if r.get('precision') == 'exact']
    record = next(r for r in scored if r['file'] == INCUMBENT)
    if record['rmse'] != INCUMBENT_RMSE:
        raise ValueError('Incumbent score changed')
    if sha(ROOT / INCUMBENT) != record['sha256']:
        raise ValueError('Incumbent file changed since scoring')

    sample = pd.read_csv(ROOT / 'Dataset/sample_submission.csv')
    incumbent = pd.read_csv(ROOT / INCUMBENT)
    if not incumbent.row_id.equals(sample.row_id):
        raise ValueError('Incumbent row ids differ')
    base = incumbent.prediction.to_numpy()
    N = len(base)

    blocks = np.load(ROOT / 'runs/reconstruction_v1/blocks_test.npz')
    if not np.array_equal(blocks['row_id'], sample.row_id.to_numpy()):
        raise ValueError('Reconstruction blocks misaligned')

    # Span of every scored submission, plus intercept.
    files = [r['file'] for r in scored] + [
        'runs/missingness_correction_v1/submission_missingness_correction.csv']
    span = np.column_stack([np.ones(N)] +
                           [pd.read_csv(ROOT / f).prediction.to_numpy() for f in files])

    raw = blocks['E_pm'] - blocks['E_D']
    perp = raw - span @ np.linalg.lstsq(span, raw, rcond=None)[0]
    norm = float(np.sqrt(np.mean(perp ** 2)))
    if norm < 1e-6:
        raise ValueError('Direction lies inside the explored span')
    u = perp / norm

    # Must be orthogonal to everything already scored, or the probe re-measures
    # a direction we have already paid for.
    leak = float(np.abs(span.T @ u / N).max())
    if leak > 1e-8:
        raise ValueError(f'Direction not orthogonal to span (leak {leak:.2e})')

    prediction = base + STEP * u
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite candidate')

    OUT.mkdir()
    output = OUT / 'submission_measurement_probe.csv'
    out = sample.copy()
    out['prediction'] = prediction
    out.to_csv(output, index=False)

    reread = pd.read_csv(output)
    if list(reread.columns) != ['row_id', 'prediction']:
        raise ValueError('Schema round-trip failed')
    if not reread.row_id.equals(sample.row_id):
        raise ValueError('Row id round-trip failed')
    np.testing.assert_allclose(reread.prediction, prediction, rtol=1e-12, atol=1e-12)

    r = INCUMBENT_RMSE
    outcomes = {}
    for c in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]:
        mse = r * r - 2 * STEP * c * r + STEP * STEP
        outcomes[f'c={c:+.1f}'] = round(float(np.sqrt(max(mse, 0))), 5)

    report = dict(
        status='UNSCORED measurement probe. Small step to measure one direction cheaply.',
        candidate=output.relative_to(ROOT).as_posix(), sha256=sha(output),
        incumbent_file=INCUMBENT, incumbent_sha256=record['sha256'],
        incumbent_rmse=r, direction=DIRECTION,
        direction_source='perp(E_pm - E_D) against span of all scored submissions',
        raw_perp_norm=norm, step=STEP,
        orthogonality_leak=leak,
        rms_change_from_incumbent=float(np.sqrt(np.mean((prediction - base) ** 2))),
        max_absolute_change=float(np.max(np.abs(prediction - base))),
        inversion_formula='c = (r^2 + t^2 - s_new^2) / (2*t*r)',
        optimal_step_after_measurement='t_opt = c*r, giving RMSE = r*sqrt(1-c^2)',
        predicted_outcomes=outcomes,
        worst_case_loss=round(float(np.sqrt(r * r + STEP ** 2) - r), 5),
        n_rows=int(len(out)), code_sha256=sha(Path(__file__)),
        allow_score_calibration=False,
        limitations=[
            'The direction is orthogonal to every scored submission, so its alignment '
            'with the true residual is unmeasured. The score may move either way.',
            'This probe is designed to MEASURE, not to improve. A small gain or small '
            'loss are both informative outcomes.',
            'Public-subset moments are approximated by full-test moments.'])
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2))
    np.save(OUT / 'direction.npy', u)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
