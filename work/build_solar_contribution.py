"""One new target-structure experiment: an independent solar contribution.

No hidden target labels exist. Sensor reconstruction and direction stability
are checked locally; sign and target coefficient require a measured score.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from model3 import build_Y
from reconstruction_repair import RepairedParams, map_latents
from deterministic_moments import moments, normal_nodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/solar_contribution_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residualize(raw, predictions, fit):
    mean = predictions[fit].mean(axis=0)
    design = predictions-mean
    u, singular, vt = np.linalg.svd(design[fit], full_matrices=False)
    # Small differences between nearly redundant submissions made the complete
    # projection unstable on one row partition. Retain substantial directions;
    # this cutoff is selected without target scores, then checked on both halves.
    keep = singular**2/np.sum(fit) > .025
    center = raw[fit].mean()
    coefficient = vt[keep].T @ ((u[:, keep].T@(raw[fit]-center))/singular[keep])
    residual = raw-center-design@coefficient
    return residual, dict(rank=int(keep.sum()), intercept=float(center-mean@coefficient),
        coefficients=coefficient.tolist(), residual_sd=float(residual[fit].std()),
        minimum_direction_energy=.025,
        singular_values=singular.tolist())


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the experiment')
    ledger = json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    sample = pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    best = ROOT/ledger['best_file']
    baseline = pd.read_csv(best)
    if not test.row_id.equals(sample.row_id) or not baseline.row_id.equals(sample.row_id):
        raise ValueError('IDs differ')
    predictions = []; sources = []
    for row in ledger['results']:
        path = ROOT/row['file']
        if 'sha256' in row and sha(path) != row['sha256']:
            raise ValueError('Scored file changed')
        frame = pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id):
            raise ValueError('Scored IDs differ')
        predictions.append(frame.prediction.to_numpy())
        sources.append(dict(file=row['file'], sha256=sha(path)))
    predictions = np.column_stack(predictions)
    z = np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    if not np.array_equal(z['row_id'], sample.row_id):
        raise ValueError('Moment IDs differ')
    solar = z['solar']
    full, projection = residualize(solar, predictions, np.ones(len(test), bool))
    split = test.row_id.to_numpy() % 2 == 0
    stability = []
    for fit in [split, ~split]:
        residual, _ = residualize(solar, predictions, fit)
        holdout = ~fit
        scale = np.std(full[holdout])
        stability.append(dict(fit_rows=int(fit.sum()), heldout_rows=int(holdout.sum()),
            heldout_correlation=float(np.corrcoef(residual[holdout], full[holdout])[0, 1]),
            heldout_relative_rms_difference=float(np.sqrt(np.mean((residual[holdout]-full[holdout])**2))/scale)))
    if any(r['heldout_correlation'] < .995 or r['heldout_relative_rms_difference'] > .03 for r in stability):
        raise ValueError('Solar direction unstable across unlabeled partitions')
    # Fresh observed solar sensors, hidden before inference. These checks concern
    # reconstruction only; they do not validate a solar coefficient in NDEM.
    excluded = set()
    for file in (ROOT/'runs').glob('*/holdout_rows.json'):
        for ids in json.loads(file.read_text()).values():
            excluded.update(ids)
    fresh = test[test.feature_12.notna() & ~test.row_id.isin(excluded)]
    audit = fresh.sample(min(6000, len(fresh)), random_state=1011).copy()
    truth = audit.feature_12.to_numpy().copy()
    audit['feature_12'] = np.nan
    original = pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired = RepairedParams.from_original(original, json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    th, cov, info = map_latents(build_Y(audit), repaired, need_cov=True, return_info=True)
    if info['failed_fallback'].any():
        raise ValueError('Failed holdout reconstruction')
    predicted = moments(th, cov, repaired, normal_nodes(8, 104), info['accepted'])['solar']
    rmse = float(np.sqrt(np.mean((predicted-truth)**2)))
    mean_rmse = float(np.std(truth))
    if rmse >= .3*mean_rmse:
        raise ValueError('Solar component reconstruction is too weak for the planned experiment')
    # One preregistered positive coefficient. A 0.20 RMS step is smaller than the
    # earlier 0.50 steps. It is an experiment, not an estimated target coefficient.
    effect = full/full.std()
    change = .20*effect
    frame = sample.copy()
    frame['prediction'] = baseline.prediction.to_numpy()+change
    if not np.isfinite(frame.prediction).all() or not frame.row_id.is_unique:
        raise ValueError('Invalid prediction')
    OUT.mkdir()
    output = OUT/'submission_solar_contribution.csv'
    frame.to_csv(output, index=False)
    roundtrip = pd.read_csv(output)
    if list(roundtrip.columns) != ['row_id', 'prediction'] or not roundtrip.row_id.equals(sample.row_id):
        raise ValueError('Round-trip schema failure')
    np.testing.assert_allclose(roundtrip.prediction, frame.prediction, rtol=1e-12, atol=1e-12)
    np.save(OUT/'direction.npy', change)
    (OUT/'holdout_rows.json').write_text(json.dumps(dict(solar=audit.row_id.tolist())))
    np.savez_compressed(OUT/'sensor_audit.npz', row_id=audit.row_id.to_numpy(), truth=truth, prediction=predicted)
    manifest = dict(status='Unscored independent solar-contribution experiment.',
        candidate=output.relative_to(ROOT).as_posix(), sha256=sha(output),
        incumbent_file=best.relative_to(ROOT).as_posix(), incumbent_sha256=sha(best),
        incumbent_rmse=ledger['best_public_rmse'], allow_score_calibration=True,
        hypothesis='The target needs a separately weighted solar contribution beyond the tied solar/wind generation term.',
        raw_effect='E[solar_generation]', rms_step=.20, raw_solar_sd=float(solar.std()),
        independent_solar_sd=float(full.std()), unrepresented_variance_fraction=float(full.var()/solar.var()),
        raw_solar_increment_coefficient=float(.20/full.std()), projection=projection, sources=sources,
        stability=stability, sensor_audit=dict(n=len(audit), rmse=rmse, constant_baseline_rmse=mean_rmse,
            r_squared=float(1-rmse**2/mean_rmse**2)), target_rmse_forecast=None,
        score_values_used_for_new_coefficient=False,
        rms_change=float(np.sqrt(np.mean(change**2))), maximum_change=float(np.max(np.abs(change))),
        zero_residual_signal_rmse_scenario=float(np.sqrt(ledger['best_public_rmse']**2+.2**2)),
        public_q_random_subset_se=float(np.sqrt(.7*np.var(change**2, ddof=1)/30000)),
        code_sha256=sha(Path(__file__)),
        limitations=['Solar sign and coefficient in the target are unverified.',
            'Orthogonality and scenario calculations use the full test data, not the unknown public subset.',
            'Sensor accuracy and geometric stability do not validate the target formula.',
            'An independent solar component is a hypothesis; its unrepresented variance is not evidence of target signal.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['projection','sources']}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
