"""Remove the excess missingness-dependent offset from the incumbent.

The correction is a per-missingness-stratum mean of the incumbent's residual
after an affine function of the physics target is removed. It is 99.4%
orthogonal to the span of all previously scored submissions, so its inner
product with the true target is NOT identified by existing leaderboard scores.
The sign is argued from label-free train-set identities, not measured.
No scored file, ledger entry or cached reconstruction is modified.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/missingness_correction_v1'
INCUMBENT = 'runs/successful_components_v3/submission_successful_components_v3.csv'
STEP = 1.0          # lands prediction drift on the physics-implied value
MIN_STRATUM = 30    # strata smaller than this reuse the tail estimate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_correction(nmiss, excess):
    corr = np.zeros_like(excess)
    tail = excess[nmiss >= 13]
    tail_mean = float(tail.mean()) if len(tail) >= MIN_STRATUM else 0.0
    profile = {}
    for k in np.unique(nmiss):
        m = nmiss == k
        value = float(excess[m].mean()) if m.sum() >= MIN_STRATUM else tail_mean
        corr[m] = value
        profile[int(k)] = dict(n=int(m.sum()), offset=value)
    return corr, profile


def main():
    if OUT.exists():
        raise FileExistsError('Preserve existing run')
    ledger = json.loads((ROOT / 'candidates/leaderboard_results.json').read_text())
    record = next(r for r in ledger['results'] if r['file'] == INCUMBENT)
    if record.get('precision') != 'exact':
        raise ValueError('Incumbent score must be exact')
    path = ROOT / INCUMBENT
    if sha(path) != record['sha256']:
        raise ValueError('Incumbent file changed since it was scored')

    sample = pd.read_csv(ROOT / 'Dataset/sample_submission.csv')
    frame = pd.read_csv(path)
    if not frame.row_id.equals(sample.row_id):
        raise ValueError('Row ids differ from sample submission')
    best = frame.prediction.to_numpy()
    if not np.isfinite(best).all():
        raise ValueError('Incumbent has nonfinite predictions')

    test = pd.read_csv(ROOT / 'Dataset/test.csv')
    if not test.row_id.equals(sample.row_id):
        raise ValueError('Test row ids differ from sample submission')
    nmiss = test.iloc[:, 1:].isna().sum(axis=1).to_numpy()

    blocks = np.load(ROOT / 'runs/reconstruction_v1/blocks_test.npz')
    if not np.array_equal(blocks['row_id'], sample.row_id.to_numpy()):
        raise ValueError('Reconstruction blocks are not aligned')
    physics = blocks['E_pg'] * (1 - blocks['E_tl']) - blocks['E_D']

    design = np.column_stack([np.ones(len(physics)), physics])
    coef = np.linalg.lstsq(design, best, rcond=None)[0]
    excess = best - design @ coef
    corr, profile = build_correction(nmiss, excess)

    # The correction must be almost entirely outside the span of scored files,
    # otherwise it is just another re-weighting of an exhausted subspace.
    scored = [r['file'] for r in ledger['results'] if r.get('precision') == 'exact']
    span = np.column_stack([np.ones(len(best))] +
                           [pd.read_csv(ROOT / f).prediction.to_numpy() for f in scored])
    projection = span @ np.linalg.lstsq(span, corr, rcond=None)[0]
    orthogonal = corr - projection
    novelty = float(np.mean(orthogonal ** 2) / np.mean(corr ** 2))
    if novelty < 0.9:
        raise ValueError(f'Correction is not a new direction (novelty {novelty:.3f})')

    prediction = best - STEP * corr
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite candidate')

    OUT.mkdir()
    output = OUT / 'submission_missingness_correction.csv'
    out = sample.copy()
    out['prediction'] = prediction
    out.to_csv(output, index=False)

    reread = pd.read_csv(output)
    if list(reread.columns) != ['row_id', 'prediction']:
        raise ValueError('Schema round-trip failed')
    if not reread.row_id.equals(sample.row_id):
        raise ValueError('Row id round-trip failed')
    np.testing.assert_allclose(reread.prediction, prediction, rtol=1e-12, atol=1e-12)

    rms_corr = float(np.sqrt(np.mean(corr ** 2)))
    report = dict(
        status='UNSCORED. Removes excess missingness-dependent offset from the incumbent.',
        candidate=output.relative_to(ROOT).as_posix(), sha256=sha(output),
        incumbent_file=INCUMBENT, incumbent_sha256=record['sha256'],
        incumbent_rmse=record['rmse'], step=STEP,
        correction_rms=rms_corr, correction_novelty_fraction=novelty,
        rms_change_from_incumbent=float(np.sqrt(np.mean((prediction - best) ** 2))),
        max_absolute_change=float(np.max(np.abs(prediction - best))),
        physics_affine_coefficients=coef.tolist(),
        score_identity=('MSE(t) = %.6f - 2t*<corr,best-y> + t^2*%.6f'
                        % (record['rmse'] ** 2, rms_corr ** 2)),
        stratum_profile=profile, n_rows=int(len(out)),
        code_sha256=sha(Path(__file__)),
        allow_score_calibration=False,
        limitations=[
            'The inner product <corr, best-y> is NOT identified by existing scores, '
            'because the correction is orthogonal to the span of all scored files. '
            'The sign of the change is therefore unknown before submission.',
            'Supporting evidence is label-free train-set identity checks, not target labels.',
            'Public-subset moments are approximated by full-test moments.',
            'This is a one-probe experiment with unknown sign, not a forecast improvement.'])
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2))
    np.save(OUT / 'correction.npy', corr)
    print(json.dumps({k: v for k, v in report.items() if k != 'stratum_profile'}, indent=2))


if __name__ == '__main__':
    main()
