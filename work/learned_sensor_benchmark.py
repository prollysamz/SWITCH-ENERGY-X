"""Train conditional sensor models without target labels or leaderboard fitting.

Training/calibration use separate train.csv rows. Evaluation uses test.csv
sensor values hidden before inference. Naturally missing values lack labels;
observed-sensor results do not establish NDEM or missing-cell accuracy.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from model3 import build_Y, derived
from reconstruction_repair import RepairedParams, map_latents, LOW, HIGH
from deterministic_moments import normal_nodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/learned_sensor_v1'
SCENARIOS = [('panel', 7, [7]), ('panel_sparse', 7, [7, 15, 2]),
             ('gross_sparse', 14, [14, 21]), ('wind_sparse', 13, [4, 13, 14, 18, 21])]


def physics(frame, params, target):
    values = []
    uncertainty = []
    for start in range(0, len(frame), 5000):
        th, cov, info = map_latents(build_Y(frame.iloc[start:start+5000]), params,
                                  need_cov=True, return_info=True)
        if info['failed_fallback'].any():
            raise ValueError('Unresolved physics fallback')
        if target == 7:
            pred = derived(th, params)[1]
            v = np.array([params.a1, 1, 0, -params.a2, 0, 0, 0, 0])
            variance = np.einsum('i,nij,j->n', v, cov, v)
        else:
            chol = np.linalg.cholesky(cov + np.eye(8)*1e-12)
            total = np.zeros(len(th)); square = np.zeros(len(th))
            nodes = normal_nodes(8, 104)
            for node in nodes:
                sample = th + np.einsum('nij,j->ni', chol, node)
                sample[:, 0] = np.clip(sample[:, 0], 0, 1300)
                sample[:, 3] = np.clip(sample[:, 3], 0, 25)
                sample[:, 5] = np.clip(sample[:, 5], 0, 1.05)
                sample[:, 6] = np.clip(sample[:, 6], .005, 1.05)
                sample[info['accepted']] = np.clip(sample[info['accepted']], LOW, HIGH)
                y = derived(sample, params)[6 if target == 14 else 5]
                total += y/len(nodes); square += y*y/len(nodes)
            pred = total; variance = np.maximum(square-total*total, 0)
        values.extend(pred); uncertainty.extend(np.sqrt(np.maximum(variance, 0)))
    return np.array(values), np.array(uncertainty)


def comparison(truth, baseline, candidate):
    diff = (truth-baseline)**2 - (truth-candidate)**2
    return dict(n=len(truth), physics_rmse=float(np.sqrt(np.mean((truth-baseline)**2))),
                learned_rmse=float(np.sqrt(np.mean((truth-candidate)**2))),
                mse_gain=float(diff.mean()), paired_se=float(diff.std(ddof=1)/np.sqrt(len(diff))))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():
        raise FileExistsError('Preserve benchmark')
    train = pd.read_csv(ROOT/'Dataset/train.csv')
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    params = RepairedParams.from_original(pickle.loads((ROOT/'work/params_final.pkl').read_bytes()),
                json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    excluded = set()
    for file in (ROOT/'runs').glob('*/holdout_rows.json'):
        for ids in json.loads(file.read_text()).values():
            excluded.update(ids)
    test = test[~test.row_id.isin(excluded)]
    results = []; saved = {}; started = time.time()
    for label, target, hidden in SCENARIOS:
        columns = [f'feature_{i:02d}' for i in range(1, 23) if i != 20 and i not in hidden]
        target_col = f'feature_{target:02d}'
        fit = train[(train.row_id % 5 != 0) & train[target_col].notna()].sample(60000, random_state=981)
        calibration = train[(train.row_id % 5 == 0) & train[target_col].notna()].sample(10000, random_state=982)
        available = test[test[target_col].notna()]
        evaluation = available.sample(min(6000, len(available)), random_state=983)
        frames = [fit.copy(), calibration.copy(), evaluation.copy()]
        ys = [f[target_col].to_numpy().copy() for f in frames]
        saved[label] = evaluation.row_id.tolist()
        for f in frames:
            for fid in hidden:
                f[f'feature_{fid:02d}'] = np.nan
        x = [f[columns].to_numpy() for f in frames]
        print(f'{label}: reconstructing hidden sensors ({time.time()-started:.1f}s)', flush=True)
        phys = [physics(f, params, target) for f in frames]
        hx = [np.column_stack([xx, p, sd]) for xx, (p, sd) in zip(x, phys)]
        models = {}; predictions = {}
        for kind in ['direct', 'residual']:
            model = HistGradientBoostingRegressor(max_iter=250, max_leaf_nodes=31,
                learning_rate=.075, l2_regularization=10, min_samples_leaf=50,
                early_stopping=True, validation_fraction=.15, n_iter_no_change=25, random_state=984)
            inputs = x if kind == 'direct' else hx
            fit_y = ys[0] if kind == 'direct' else ys[0]-phys[0][0]
            model.fit(inputs[0], fit_y)
            guesses = [model.predict(xx) for xx in inputs[1:]]
            if kind == 'residual':
                guesses = [g + p[0] for g, p in zip(guesses, phys[1:])]
            models[kind] = model; predictions[kind] = guesses
        # Choose model and shrinkage using train calibration only, before test evaluation.
        options = []
        for kind, (cal_pred, _) in predictions.items():
            change = cal_pred - phys[1][0]
            weight = float(np.clip(np.dot(ys[1]-phys[1][0], change)/max(np.dot(change, change), 1e-15), 0, 1))
            mse = np.mean((ys[1]-phys[1][0]-weight*change)**2)
            options.append((mse, kind, weight))
        _, selected, weight = min(options)
        final = phys[2][0] + weight*(predictions[selected][1]-phys[2][0])
        record = dict(scenario=label, hidden=hidden, training_rows=len(fit), calibration_rows=len(calibration),
            selected=selected, calibration_weight=weight, **comparison(ys[2], phys[2][0], final),
            direct=comparison(ys[2], phys[2][0], predictions['direct'][1]),
            residual=comparison(ys[2], phys[2][0], predictions['residual'][1]))
        results.append(record)
        (OUT/f'{label}_models.pkl').write_bytes(pickle.dumps(dict(models=models, columns=columns,
            target=target, hidden=hidden, selected=selected, weight=weight)))
        np.savez_compressed(OUT/f'{label}_evaluation.npz', row_id=evaluation.row_id.to_numpy(),
            truth=ys[2], baseline=phys[2][0], prediction=final,
            direct=predictions['direct'][1], residual=predictions['residual'][1])
        (OUT/'progress.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
    report = dict(results=results, elapsed_seconds=time.time()-started,
        target_weights='Untouched; no NDEM predictions fit or scored.',
        split='Train rows fit/calibrate; previously unused test rows evaluated after selection.',
        limitation='Observed-sensor holdouts do not measure naturally missing cells or hidden target accuracy.')
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    (OUT/'holdout_rows.json').write_text(json.dumps(saved))


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
