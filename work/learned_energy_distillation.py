"""Self-supervised correction of incomplete-row energy estimates.

Pseudo-labels are frozen-model estimates from richer sensor observations,
NOT hidden NDEM labels. No leaderboard score enters the learned fit. Row splits
are disjoint and exclude rows used to estimate physics parameters.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scipy.optimize import least_squares
from model3 import build_Y, map_latents as old_map, resid_fn, SIG, PRIOR_SD
from reconstruction_repair import RepairedParams, map_latents, informed_start, LOW, HIGH, robust_cost
from deterministic_moments import moments, normal_nodes
from frozen_current_target import predict
from learned_sensor_benchmark import comparison

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/learned_energy_v1'
COLS = [f'feature_{i:02d}' for i in range(1, 23) if i != 20]


def stable_map(y, repaired):
    try:
        return map_latents(y, repaired, need_cov=True, return_info=True)
    except RuntimeError:
        # Artificial masking can create harder problems than the natural data.
        # Preserve every audit row; isolate the failure and require convergence.
        if len(y) > 1:
            mid = len(y)//2
            a = stable_map(y[:mid], repaired); b = stable_map(y[mid:], repaired)
            return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), {
                k:np.concatenate([a[2][k], b[2][k]]) for k in ['accepted', 'failed_fallback']}
        observed = np.isfinite(y)
        def residual(x):
            return resid_fn(x[None, :], np.nan_to_num(y), observed, 1/SIG, repaired)[0]
        fits = []
        for wind in [None, 3., 9., 18.]:
            start = informed_start(y[0], repaired)
            if wind is not None:
                start[3] = wind
            fit = least_squares(residual, start, bounds=(LOW, HIGH), loss='soft_l1', f_scale=2.5,
                x_scale=PRIOR_SD, max_nfev=2000, ftol=1e-6, xtol=1e-6, gtol=1e-6)
            if fit.success and np.isfinite(fit.x).all():
                fits.append((robust_cost(residual(fit.x)), fit))
        if not fits:
            raise RuntimeError('Extended masked-row solver failed; audit cannot proceed')
        fit = min(fits, key=lambda z:z[0])[1]
        cov = np.linalg.inv(fit.jac.T@fit.jac + np.diag(1e-8/PRIOR_SD**2))
        print('One artificially masked row required the extended converged solver.', flush=True)
        return fit.x[None, :], cov[None, :, :], {'accepted':np.array([True]), 'failed_fallback':np.array([False])}


def reconstruct(frame, original, repaired, spec):
    target = []; extras = []
    for a in range(0, len(frame), 5000):
        y = build_Y(frame.iloc[a:a+5000])
        th0, cov0 = old_map(y, original, need_cov=True)
        th, cov, info = stable_map(y, repaired)
        if info['failed_fallback'].any():
            raise ValueError('Unresolved fallback')
        z0 = moments(th0, cov0, original, normal_nodes(8, 104))
        z = moments(th, cov, repaired, normal_nodes(8, 104), info['accepted'])
        f = predict(z0, z, z['pg_panel'], spec)
        target.extend(f)
        extras.append(np.column_stack([f, th, np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))]))
    return np.array(target), np.concatenate(extras)


def masked_view(frame, seed):
    rng = np.random.default_rng(seed)
    masked = frame.copy()
    # Random outages exercise natural combinations; structured views ensure
    # rare information-poor regimes have enough training examples.
    group = rng.choice(4, len(frame), p=[.5, .2, .2, .1])
    random_mask = rng.random((len(frame), len(COLS))) < rng.uniform(.04, .22, (len(frame), 1))
    values = masked[COLS].to_numpy().copy()
    values[random_mask & (group[:, None] == 0)] = np.nan
    for g, hidden in [(1, [7, 15, 2]), (2, [4, 13, 14, 18, 21])]:
        for fid in hidden:
            values[group == g, COLS.index(f'feature_{fid:02d}')] = np.nan
    masked[COLS] = values
    return masked, group


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():
        raise FileExistsError('Preserve result')
    started = time.time()
    train = pd.read_csv(ROOT/'Dataset/train.csv')
    physics_ids = train.sample(120000, random_state=11).row_id
    train = train[~train.row_id.isin(physics_ids)]
    parts = []
    for folds, size in [([2, 3, 4], 60000), ([1], 15000), ([0], 15000)]:
        eligible = train[train.row_id.mod(5).isin(folds)]
        parts.append(eligible.sample(size, random_state=991))
    frame = pd.concat(parts, ignore_index=True)
    assert frame.row_id.is_unique
    original = pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired = RepairedParams.from_original(original, json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    spec = json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    cached = OUT/'training_arrays.npz'
    if cached.exists():
        z = np.load(cached)
        if not np.array_equal(z['row_id'], frame.row_id):
            raise ValueError('Cache IDs differ')
        teacher, base, inputs, groups = [z[k] for k in ['teacher', 'base', 'inputs', 'groups']]
    else:
        print('Computing richer-observation teacher', flush=True)
        teacher, _ = reconstruct(frame, original, repaired, spec)
        masked, groups = masked_view(frame, 992)
        print(f'Computing incomplete-observation baseline ({time.time()-started:.1f}s)', flush=True)
        base, extras = reconstruct(masked, original, repaired, spec)
        inputs = np.column_stack([masked[COLS].to_numpy(), extras])
        np.savez_compressed(cached, row_id=frame.row_id.to_numpy(), teacher=teacher, base=base,
                            inputs=inputs, groups=groups)
    fit = np.arange(60000); cal = np.arange(60000, 75000); audit = np.arange(75000, 90000)
    model = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=31, learning_rate=.06,
        l2_regularization=30, min_samples_leaf=100, early_stopping=True, validation_fraction=.15,
        n_iter_no_change=25, random_state=993)
    model.fit(inputs[fit], teacher[fit]-base[fit])
    correction = model.predict(inputs)
    weights = {}
    # Selection by actual missingness, never by the synthetic augmentation label.
    # This same observable partition can be used on real test rows.
    raw = inputs[:, :len(COLS)]
    wind = np.all(np.isnan(raw[:, [COLS.index(f'feature_{i:02d}') for i in [4, 13, 14, 18, 21]]]), axis=1)
    thermal = np.all(np.isnan(raw[:, [COLS.index(f'feature_{i:02d}') for i in [7, 15, 2]]]), axis=1)
    regime = np.where(wind, 'wind', np.where(thermal, 'thermal', 'other'))
    candidate = base.copy()
    for label in ['wind', 'thermal', 'other']:
        c = cal[regime[cal] == label]
        d = correction[c]
        weight = float(np.clip(np.dot(teacher[c]-base[c], d)/max(np.dot(d, d), 1e-15), 0, 1))
        weights[label] = weight
        candidate[regime == label] += weight*correction[regime == label]
    results = []
    for label in ['all', 'wind', 'thermal', 'other']:
        ix = audit if label == 'all' else audit[regime[audit] == label]
        results.append(dict(regime=label, **comparison(teacher[ix], base[ix], candidate[ix])))
    # Restore natural test-regime prevalence: augmentation intentionally
    # oversamples sparse wind cases, which must not inflate the overall gain.
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    tw = test[[f'feature_{i:02d}' for i in [4, 13, 14, 18, 21]]].isna().all(axis=1)
    tt = test[[f'feature_{i:02d}' for i in [7, 15, 2]]].isna().all(axis=1)
    proportions = dict(wind=float(tw.mean()), thermal=float((tt & ~tw).mean()),
                       other=float((~tt & ~tw).mean()))
    natural_gain = sum(proportions[r['regime']]*r['mse_gain'] for r in results[1:])
    natural_se = np.sqrt(sum((proportions[r['regime']]*r['paired_se'])**2 for r in results[1:]))
    natural_base = np.sqrt(sum(proportions[r['regime']]*r['physics_rmse']**2 for r in results[1:]))
    natural_candidate = np.sqrt(sum(proportions[r['regime']]*r['learned_rmse']**2 for r in results[1:]))
    # Nothing is selected using these audit results; they only accept/reject.
    gate = natural_gain > 3*natural_se and all(
        r['mse_gain'] >= -2*r['paired_se'] for r in results[1:])
    report = dict(status='Self-supervised pseudo-target evaluation, not hidden NDEM validation.',
        results=results, calibration_weights=weights, gate=bool(gate), n_iterations=model.n_iter_,
        natural_prevalence_weighted=dict(proportions=proportions, mse_gain=float(natural_gain),
            paired_se=float(natural_se), baseline_rmse=float(natural_base), learned_rmse=float(natural_candidate)),
        teacher='Frozen 0.60058 functional predictor on richer observations of the same training row.',
        target_weights_changed=False, leaderboard_scores_used_for_learning=False,
        limitations=['Teacher inherits existing target-formula errors; cannot discover missing target terms.',
            'Artificial outages and naturally missing values may have different conditional distributions.',
            'Reported RMSE measures teacher recovery, not competition target accuracy.'],
        elapsed_seconds=time.time()-started)
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    (OUT/'model.pkl').write_bytes(pickle.dumps(dict(model=model, columns=COLS, weights=weights)))
    (OUT/'provenance.json').write_text(json.dumps(dict(
        physics_parameter_rows_excluded=True, train_ids=parts[0].row_id.tolist(),
        calibration_ids=parts[1].row_id.tolist(), audit_ids=parts[2].row_id.tolist(),
        frozen_spec_sha256=hashlib.sha256((ROOT/'runs/deterministic_v1/frozen_target.json').read_bytes()).hexdigest())))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
