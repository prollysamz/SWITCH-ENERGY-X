"""Confirm the frozen learned correction on a second disjoint row sample."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from learned_energy_distillation import ROOT, OUT, COLS, reconstruct, masked_view
from learned_sensor_benchmark import comparison
from reconstruction_repair import RepairedParams


def regimes(raw):
    wind = np.isnan(raw[:, [COLS.index(f'feature_{i:02d}') for i in [4, 13, 14, 18, 21]]]).all(axis=1)
    thermal = np.isnan(raw[:, [COLS.index(f'feature_{i:02d}') for i in [7, 15, 2]]]).all(axis=1)
    return np.where(wind, 'wind', np.where(thermal, 'thermal', 'other'))


def weighted_report(results, proportions):
    gain = sum(proportions[r['regime']]*r['mse_gain'] for r in results)
    se = np.sqrt(sum((proportions[r['regime']]*r['paired_se'])**2 for r in results))
    base = np.sqrt(sum(proportions[r['regime']]*r['physics_rmse']**2 for r in results))
    new = np.sqrt(sum(proportions[r['regime']]*r['learned_rmse']**2 for r in results))
    return dict(baseline_rmse=float(base), learned_rmse=float(new), mse_gain=float(gain),
                paired_se=float(se), gate=bool(gain > 3*se and all(r['mse_gain'] >= -2*r['paired_se'] for r in results)))


def main():
    if (OUT/'confirmation.json').exists():
        raise FileExistsError('Preserve confirmation')
    run = json.loads((OUT/'report.json').read_text())
    state = pickle.loads((OUT/'model.pkl').read_bytes())
    provenance = json.loads((OUT/'provenance.json').read_text())
    train = pd.read_csv(ROOT/'Dataset/train.csv')
    excluded = set(train.sample(120000, random_state=11).row_id)
    for key in ['train_ids', 'calibration_ids', 'audit_ids']:
        excluded.update(provenance[key])
    frame = train[~train.row_id.isin(excluded)].sample(30000, random_state=997).reset_index(drop=True)
    original = pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired = RepairedParams.from_original(original, json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    spec = json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    teacher, _ = reconstruct(frame, original, repaired, spec)
    masked, _ = masked_view(frame, 998)
    base, extra = reconstruct(masked, original, repaired, spec)
    raw = masked[COLS].to_numpy()
    inputs = np.column_stack([raw, extra])
    group = regimes(raw)
    delta = state['model'].predict(inputs)
    result = base + np.array([state['weights'][g] for g in group])*delta
    results = [dict(regime=g, **comparison(teacher[group == g], base[group == g], result[group == g]))
               for g in ['wind', 'thermal', 'other']]
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    tg = regimes(test[COLS].to_numpy())
    proportions = {g:float(np.mean(tg == g)) for g in ['wind', 'thermal', 'other']}
    first = weighted_report(run['results'][1:], proportions)
    second = weighted_report(results, proportions)
    report = dict(status='Second disjoint pseudo-target audit; no hidden target labels.',
        first_audit_weighted=first, confirmation_weighted=second, results=results,
        natural_test_proportions=proportions, gate=bool(first['gate'] and second['gate']),
        model_sha256=hashlib.sha256((OUT/'model.pkl').read_bytes()).hexdigest(),
        limitation='Within-regime artificial-outage patterns can still differ from natural missingness.')
    (OUT/'confirmation.json').write_text(json.dumps(report, indent=2))
    (OUT/'confirmation_rows.json').write_text(json.dumps(frame.row_id.tolist()))
    np.savez_compressed(OUT/'confirmation_predictions.npz', row_id=frame.row_id.to_numpy(),
                        teacher=teacher, baseline=base, prediction=result, regime=group)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
