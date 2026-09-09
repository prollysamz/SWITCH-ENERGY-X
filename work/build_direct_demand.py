"""Test a direct demand penalty beyond battery-coupled demand terms.

The scored solar direction and incumbent are protected during residualization.
Sensor tests concern imputation, not the hidden NDEM coefficient.
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
OUT = ROOT/'runs/direct_demand_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residualize(raw, predictions, protected, fit):
    pmean = protected[fit].mean(axis=0)
    protected_centered = protected-pmean
    mean = predictions[fit].mean(axis=0)
    pc = np.linalg.lstsq(protected_centered[fit], predictions[fit]-mean, rcond=None)[0]
    remainder = predictions-mean-protected_centered@pc
    _, singular, vt = np.linalg.svd(remainder[fit], full_matrices=False)
    keep = singular**2/np.sum(fit) > .025
    design = np.column_stack([protected_centered, remainder@vt[keep].T])
    center = raw[fit].mean()
    coefficient = np.linalg.lstsq(design[fit], raw[fit]-center, rcond=None)[0]
    result = raw-center-design@coefficient
    return result, dict(protected_mean=pmean.tolist(), prediction_mean=mean.tolist(),
        protected_projection=pc.tolist(), retained_vectors=vt[keep].tolist(),
        coefficients=coefficient.tolist(), raw_mean=float(center),
        rank=int(np.linalg.matrix_rank(design[fit])), minimum_direction_energy=.025)


def main():
    if OUT.exists():
        raise FileExistsError('Preserve experiment')
    ledger = json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    bestpath = ROOT/ledger['best_file']
    baseline = pd.read_csv(bestpath)
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    sample = pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if not baseline.row_id.equals(sample.row_id) or not test.row_id.equals(sample.row_id):
        raise ValueError('IDs differ')
    predictions = []; sources = []
    for record in ledger['results']:
        path = ROOT/record['file']
        if 'sha256' in record and sha(path) != record['sha256']:
            raise ValueError('Scored file changed')
        frame = pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id) or not np.isfinite(frame.prediction).all():
            raise ValueError('Invalid scored file')
        predictions.append(frame.prediction.to_numpy())
        sources.append(dict(file=record['file'], sha256=sha(path)))
    predictions = np.column_stack(predictions)
    solar_manifest = json.loads((ROOT/'runs/solar_contribution_v1/manifest.json').read_text())
    solar_file = ROOT/solar_manifest['candidate']
    solar_base = ROOT/solar_manifest['incumbent_file']
    if sha(solar_file) != solar_manifest['sha256'] or sha(solar_base) != solar_manifest['incumbent_sha256']:
        raise ValueError('Solar source changed')
    solar_direction = pd.read_csv(solar_file).prediction.to_numpy()-pd.read_csv(solar_base).prediction.to_numpy()
    protected = np.column_stack([baseline.prediction.to_numpy(), solar_direction])
    cache = np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    if not np.array_equal(cache['row_id'], sample.row_id):
        raise ValueError('Cache IDs differ')
    raw = -cache['E_D']
    residual, projection = residualize(raw, predictions, protected, np.ones(len(test), bool))
    stability = []
    split = test.row_id.to_numpy()%2 == 0
    for fit in [split, ~split]:
        part, _ = residualize(raw, predictions, protected, fit)
        held = ~fit
        stability.append(dict(heldout_rows=int(held.sum()),
            correlation=float(np.corrcoef(part[held], residual[held])[0, 1]),
            relative_rms_difference=float(np.sqrt(np.mean((part[held]-residual[held])**2))/residual[held].std())))
    if any(r['correlation'] < .995 or r['relative_rms_difference'] > .03 for r in stability):
        raise ValueError('Direction is unstable')
    excluded = set()
    for file in (ROOT/'runs').glob('*/holdout_rows.json'):
        for ids in json.loads(file.read_text()).values():
            excluded.update(ids)
    fresh = test[test.feature_09.notna() & ~test.row_id.isin(excluded)]
    audit = fresh.sample(min(6000, len(fresh)), random_state=1021).copy()
    truth = audit.feature_09.to_numpy().copy()
    audit['feature_09'] = np.nan
    original = pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    params = RepairedParams.from_original(original, json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    th, cov, info = map_latents(build_Y(audit), params, need_cov=True, return_info=True)
    if info['failed_fallback'].any():
        raise ValueError('Failed demand reconstruction')
    predicted = moments(th, cov, params, normal_nodes(8, 104), info['accepted'])['E_D']
    rmse = float(np.sqrt(np.mean((predicted-truth)**2)))
    prior_rmse = float(np.std(truth))
    if rmse > .5*prior_rmse:
        raise ValueError('Demand reconstruction is too weak')
    delta = .20*residual/residual.std()
    correlations = [float(np.corrcoef(delta, protected[:, i])[0, 1]) for i in range(2)]
    if max(abs(c) for c in correlations) > 1e-8:
        raise ValueError('Direction changes protected solar/incumbent components')
    candidate = sample.copy()
    candidate['prediction'] = baseline.prediction.to_numpy()+delta
    if not candidate.row_id.is_unique or not np.isfinite(candidate.prediction).all():
        raise ValueError('Invalid output')
    OUT.mkdir()
    path = OUT/'submission_direct_demand.csv'
    candidate.to_csv(path, index=False)
    reread = pd.read_csv(path)
    if list(reread.columns) != ['row_id', 'prediction'] or not reread.row_id.equals(sample.row_id):
        raise ValueError('Round-trip schema failed')
    np.testing.assert_allclose(reread.prediction, candidate.prediction, rtol=1e-12, atol=1e-12)
    np.save(OUT/'direction.npy', delta)
    np.savez_compressed(OUT/'sensor_audit.npz', row_id=audit.row_id.to_numpy(), truth=truth, prediction=predicted)
    (OUT/'holdout_rows.json').write_text(json.dumps(dict(demand=audit.row_id.tolist())))
    manifest = dict(status='Unscored direct-demand experiment.', candidate=path.relative_to(ROOT).as_posix(),
        sha256=sha(path), incumbent_file=bestpath.relative_to(ROOT).as_posix(), incumbent_sha256=sha(bestpath),
        incumbent_rmse=ledger['best_public_rmse'], allow_score_calibration=True,
        hypothesis='An additional negative linear demand contribution is needed beyond battery-coupled demand effects.',
        raw_effect='-E[demand]', effect_sd=.20, raw_sd=float(raw.std()), residual_sd=float(residual.std()),
        independent_variance_fraction=float(residual.var()/raw.var()),
        raw_demand_penalty_increment=float(.20/residual.std()), projection=projection, stability=stability,
        protected_correlations=correlations, sources=sources, rows=len(candidate),
        sensor_audit=dict(rows=len(audit), rmse=rmse, constant_baseline_rmse=prior_rmse,
            r_squared=float(1-rmse**2/prior_rmse**2)),
        rms_change=float(np.sqrt(np.mean(delta**2))), maximum_absolute_change=float(np.max(np.abs(delta))),
        q_random_public_subset_se=float(np.sqrt(.7*np.var(delta**2, ddof=1)/30000)),
        new_target_score_forecast=None, new_coefficient_selected_using_scores=False,
        code_sha256=sha(Path(__file__)),
        limitations=['Demand reconstruction does not validate the target coefficient.',
            'Direction protection holds on full-test moments; the public subset is unknown.',
            'The proposed sign and 0.20 step are experimental; the next measured score may worsen.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['projection', 'sources']}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
