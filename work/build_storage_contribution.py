"""One battery-state experiment beyond existing ratio and demand effects."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from build_direct_demand import residualize, sha, ROOT
from model3 import build_Y
from reconstruction_repair import RepairedParams, map_latents
from deterministic_moments import moments, normal_nodes

OUT = ROOT/'runs/storage_contribution_v1'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve experiment')
    ledger = json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    sample = pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    bestpath = ROOT/ledger['best_file']
    best = pd.read_csv(bestpath)
    if not sample.row_id.equals(test.row_id) or not best.row_id.equals(test.row_id):
        raise ValueError('ID mismatch')
    columns = []; sources = []
    for record in ledger['results']:
        path = ROOT/record['file']; digest = sha(path)
        if 'sha256' in record and digest != record['sha256']:
            raise ValueError('A scored source changed')
        f = pd.read_csv(path)
        if not f.row_id.equals(sample.row_id) or not np.isfinite(f.prediction).all():
            raise ValueError('Invalid scored source')
        columns.append(f.prediction.to_numpy())
        sources.append(dict(file=record['file'], sha256=digest))
    predictions = np.column_stack(columns)
    protected = [best.prediction.to_numpy()]
    for name in ['solar_contribution_v1', 'direct_demand_v1']:
        manifest = json.loads((ROOT/'runs'/name/'manifest.json').read_text())
        a, b = ROOT/manifest['candidate'], ROOT/manifest['incumbent_file']
        if sha(a) != manifest['sha256'] or sha(b) != manifest['incumbent_sha256']:
            raise ValueError('Protected direction source changed')
        protected.append(pd.read_csv(a).prediction.to_numpy()-pd.read_csv(b).prediction.to_numpy())
    protected = np.column_stack(protected)
    cache = np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    if not np.array_equal(cache['row_id'], test.row_id):
        raise ValueError('Moment IDs differ')
    raw = cache['E_S']
    residual, projection = residualize(raw, predictions, protected, np.ones(len(test), bool))
    stability = []
    split = test.row_id.to_numpy()%2 == 0
    for fit in [split, ~split]:
        alternate, _ = residualize(raw, predictions, protected, fit)
        held = ~fit
        stability.append(dict(n=int(held.sum()), correlation=float(np.corrcoef(alternate[held], residual[held])[0, 1]),
            relative_rms_difference=float(np.sqrt(np.mean((alternate[held]-residual[held])**2))/residual[held].std())))
    if any(r['correlation'] < .995 or r['relative_rms_difference'] > .03 for r in stability):
        raise ValueError('Unstable new component')
    excluded = set(test[test.feature_08.notna()].sample(6000, random_state=1031).row_id)
    for file in (ROOT/'runs').glob('*/holdout_rows.json'):
        for ids in json.loads(file.read_text()).values():
            excluded.update(ids)
    eligible = test[test.feature_08.notna() & ~test.row_id.isin(excluded)]
    audit = eligible.sample(min(6000, len(eligible)), random_state=1032).copy()
    truth = audit.feature_08.to_numpy().copy(); audit['feature_08'] = np.nan
    original = pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    params = RepairedParams.from_original(original, json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    th, cov, info = map_latents(build_Y(audit), params, need_cov=True, return_info=True)
    if info['failed_fallback'].any():
        raise ValueError('Unresolved sensor fit')
    pred = moments(th, cov, params, normal_nodes(8, 104), info['accepted'])['E_S']
    rmse = float(np.sqrt(np.mean((pred-truth)**2))); prior = float(truth.std())
    if rmse >= .6*prior:
        raise ValueError('Insufficient storage reconstruction')
    delta = .20*residual/residual.std()
    correlations = [float(np.corrcoef(delta, protected[:, k])[0, 1]) for k in range(3)]
    if max(abs(v) for v in correlations) > 1e-8:
        raise ValueError('Protected target contributions are not independent')
    result = sample.copy(); result['prediction'] = best.prediction.to_numpy()+delta
    if not np.isfinite(result.prediction).all() or not result.row_id.is_unique:
        raise ValueError('Invalid predictions')
    OUT.mkdir()
    path = OUT/'submission_storage_contribution.csv'
    result.to_csv(path, index=False)
    reread = pd.read_csv(path)
    if list(reread.columns) != ['row_id', 'prediction'] or not reread.row_id.equals(test.row_id):
        raise ValueError('Round-trip schema failure')
    np.testing.assert_allclose(reread.prediction, result.prediction, rtol=1e-12, atol=1e-12)
    np.save(OUT/'direction.npy', delta)
    np.savez_compressed(OUT/'sensor_audit.npz', row_id=audit.row_id.to_numpy(), truth=truth, prediction=pred)
    (OUT/'holdout_rows.json').write_text(json.dumps(dict(state_of_charge=audit.row_id.tolist())))
    report = dict(status='Unscored independent battery-state contribution.',
        candidate=path.relative_to(ROOT).as_posix(), sha256=sha(path),
        incumbent_file=bestpath.relative_to(ROOT).as_posix(), incumbent_sha256=sha(bestpath),
        incumbent_rmse=ledger['best_public_rmse'], allow_score_calibration=True,
        hypothesis='Available stored energy has a positive direct contribution beyond demand/storage ratio and other represented effects.',
        raw_effect='E[battery_state_of_charge]', effect_sd=.20,
        raw_increment_coefficient=float(.20/residual.std()), raw_sd=float(raw.std()), residual_sd=float(residual.std()),
        independent_variance_fraction=float(residual.var()/raw.var()),
        projection=projection, sources=sources, protected_correlations=correlations, stability=stability,
        sensor_audit=dict(rows=len(audit), rmse=rmse, constant_baseline_rmse=prior, r_squared=float(1-rmse**2/prior**2)),
        n_rows=len(result), rms_change=float(np.sqrt(np.mean(delta**2))), maximum_absolute_change=float(np.max(np.abs(delta))),
        new_target_rmse_forecast=None, new_coefficient_selected_using_scores=False,
        q_random_public_subset_se=float(np.sqrt(.7*np.var(delta**2, ddof=1)/30000)),
        code_sha256={Path(__file__).name:sha(Path(__file__)), 'build_direct_demand.py':sha(ROOT/'work/build_direct_demand.py')},
        limitations=['The positive storage coefficient is a target hypothesis, not locally validated.',
            'Battery sensor accuracy is weaker than solar sensor accuracy.',
            'Protected-direction orthogonality uses full-test moments; public moments can differ.',
            'Observed-sensor holdouts do not measure hidden NDEM accuracy.'])
    (OUT/'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources', 'projection', 'code_sha256']}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
