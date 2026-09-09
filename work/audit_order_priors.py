"""Disjoint sensor audit of row-order-dependent storage and demand priors.

No target scores enter estimation or evaluation. Neighborhood estimates exclude
every audit/calibration row and its sensors. Existing model files are read-only.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits
from model3 import build_Y, PRIOR_MU, PRIOR_SD
from reconstruction_repair import RepairedParams
from learned_energy_distillation import stable_map

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/order_prior_audit_v1'


def smooth(values, eligible, width):
    ok = eligible & np.isfinite(values)
    num = uniform_filter1d(np.where(ok, values, 0.), size=width, mode='constant')
    den = uniform_filter1d(ok.astype(float), size=width, mode='constant')
    return num / np.maximum(den, 1e-12)


def update(mean, covariance, local, sd, p):
    # Replace the two Gaussian prior factors in the existing Laplace posterior.
    A = np.zeros((2, 8)); A[0, 6] = 1.; A[1, 7] = 1.; A[1, 1] = -p.d1
    old_mu = np.array([PRIOR_MU[6], p.d0])
    old_var = np.array([PRIOR_SD[6]**2, p.dsd**2])
    precision = np.linalg.inv(covariance)
    natural = np.einsum('nij,nj->ni', precision, mean)
    precision += A.T @ np.diag(1/sd**2-1/old_var) @ A
    natural += (local/sd**2-old_mu/old_var) @ A
    new_cov = np.linalg.inv(precision)
    new_mean = np.einsum('nij,nj->ni', new_cov, natural)
    return new_mean, new_cov


def compare(truth, base, candidate):
    paired = (truth-base)**2-(truth-candidate)**2
    return dict(n=len(truth), baseline_rmse=float(np.sqrt(np.mean((truth-base)**2))),
        candidate_rmse=float(np.sqrt(np.mean((truth-candidate)**2))),
        mse_gain=float(paired.mean()), paired_se=float(paired.std(ddof=1)/np.sqrt(len(truth))))


def main():
    if OUT.exists(): raise FileExistsError('Preserve existing audit')
    data = pd.concat([pd.read_csv(ROOT/'Dataset/train.csv'),
                      pd.read_csv(ROOT/'Dataset/test.csv')], ignore_index=True)
    if not np.array_equal(data.row_id, np.arange(1, len(data)+1)):
        raise ValueError('Order assumption does not match IDs')
    p = RepairedParams.from_original(pickle.loads((ROOT/'work/params_final.pkl').read_bytes()),
        json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    # Test-only audit folds, excluded from prior estimation in their entirety.
    fit = data.row_id.to_numpy()%5 != 0
    cal = (data.row_id.to_numpy()%10 == 0) & (data.row_id.to_numpy()>400000)
    audit = (data.row_id.to_numpy()%10 == 5) & (data.row_id.to_numpy()>400000)
    soc = data.feature_08.to_numpy()
    demand_residual = (data.feature_09-p.d1*data.feature_02).to_numpy()
    values = [soc, demand_residual]
    choices = []; local = []; scales = []
    for name, value in zip(['storage', 'demand_residual'], values):
        rows = cal & np.isfinite(value)
        options = []
        for width in [251, 1001, 4001, 16001]:
            estimate = smooth(value, fit, width)
            loss = float(np.mean((value[rows]-estimate[rows])**2))
            options.append((loss, width))
        _, width = min(options)
        estimate = smooth(value, fit, width)
        rows_audit = audit & np.isfinite(value)
        fixed = PRIOR_MU[6] if name == 'storage' else p.d0
        measured_sd = float(np.sqrt(np.mean((value[rows]-estimate[rows])**2)))
        choices.append(dict(name=name, width=width, calibration_options=options,
            residual_sd=measured_sd,
            heldout_prior_comparison=compare(value[rows_audit],
                np.full(rows_audit.sum(), fixed), estimate[rows_audit])))
        local.append(estimate); scales.append(measured_sd)
    local = np.column_stack(local); scales = np.array(scales)
    results = []; saved = {}; arrays = {}
    for label, target, hidden in [('storage',8,[8]), ('storage_sparse',8,[8,11,19]),
                                  ('demand',9,[9]), ('demand_sparse',9,[9,19,21])]:
        eligible = np.flatnonzero(audit & data[f'feature_{target:02d}'].notna().to_numpy())
        ids = np.random.default_rng(3011+target).choice(eligible, min(4000,len(eligible)), replace=False)
        frame = data.iloc[ids].copy()
        truth = frame[f'feature_{target:02d}'].to_numpy().copy()
        for fid in hidden: frame[f'feature_{fid:02d}'] = np.nan
        mean, cov, info = stable_map(build_Y(frame), p)
        new_mean, new_cov = update(mean, cov, local[ids], scales, p)
        index = 6 if target == 8 else 7
        record = dict(scenario=label, hidden=hidden,
            **compare(truth,mean[:,index],new_mean[:,index]))
        results.append(record); saved[label] = frame.row_id.tolist()
        arrays[label+'_truth'] = truth; arrays[label+'_baseline'] = mean[:,index]
        arrays[label+'_candidate'] = new_mean[:,index]
        print(json.dumps(record), flush=True)
    OUT.mkdir()
    report = dict(status='Independent row-order prior sensor audit; no target validation',
        prior_choices=choices, results=results,
        limitation='Laplace prior replacement approximates a nonlinear posterior. Sensor gains do not establish NDEM gains.',
        all_audit_rows_excluded_from_prior_estimation=True,
        target_scores_used=False)
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    (OUT/'holdout_rows.json').write_text(json.dumps(saved))
    np.savez_compressed(OUT/'sensor_results.npz', **arrays)
    np.savez_compressed(OUT/'test_priors.npz', row_id=data.row_id.to_numpy()[400000:],
        local=local[400000:], sd=scales)
    print(json.dumps(choices), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=6): main()
