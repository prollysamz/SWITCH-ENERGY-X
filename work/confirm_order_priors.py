"""Locked-prior confirmation and target-sensitivity diagnostic; no score tuning."""
import json
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_order_priors import ROOT, update, compare
from model3 import build_Y
from reconstruction_repair import RepairedParams
from learned_energy_distillation import stable_map

OUT = ROOT/'runs/order_prior_confirmation_v1'

def main():
    if OUT.exists(): raise FileExistsError('Preserve run')
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    prior = np.load(ROOT/'runs/order_prior_audit_v1/test_priors.npz')
    assert np.array_equal(test.row_id, prior['row_id'])
    old_rows = json.loads((ROOT/'runs/order_prior_audit_v1/holdout_rows.json').read_text())
    p = RepairedParams.from_original(pickle.loads((ROOT/'work/params_final.pkl').read_bytes()),
        json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    results=[]; saved={}
    for label,target,hidden in [('storage',8,[8]),('storage_sparse',8,[8,11,19]),
                                ('demand',9,[9]),('demand_sparse',9,[9,19,21])]:
        available = test[(test.row_id%10==5) & test[f'feature_{target:02d}'].notna()
                         & ~test.row_id.isin(old_rows[label])]
        frame=available.sample(min(2000,len(available)),random_state=90511+target).copy()
        ids=frame.index.to_numpy(); truth=frame[f'feature_{target:02d}'].to_numpy().copy()
        for fid in hidden: frame[f'feature_{fid:02d}']=np.nan
        mean,cov,info=stable_map(build_Y(frame),p)
        new,newcov=update(mean,cov,prior['local'][ids],prior['sd'],p)
        k=6 if target==8 else 7
        record=dict(scenario=label,**compare(truth,mean[:,k],new[:,k]))
        results.append(record); saved[label]=frame.row_id.tolist()
        print(json.dumps(record),flush=True)
    z=np.load(ROOT/'runs/deterministic_v1/posterior_repaired.npz')
    risk=np.load(ROOT/'runs/deterministic_v1/posterior_risk.npz')
    assert np.array_equal(z['row_id'],prior['row_id'])
    assert np.array_equal(z['row_id'],risk['row_id'])
    mean,cov=update(z['mean'],z['covariance'],prior['local'],prior['sd'],p)
    shift=mean-z['mean']
    gradient=risk['gradient']
    approximate_change=np.sum(gradient*shift,axis=1)
    variance_before=np.einsum('ni,nij,nj->n',gradient,z['covariance'],gradient)
    variance_after=np.einsum('ni,nij,nj->n',gradient,cov,gradient)
    diagnostic=dict(latent_rms_changes=dict(zip(['G','T','H','V','P','C','S','D'],
        np.sqrt(np.mean(shift**2,axis=0)).tolist())),
        old_formula_linearized_prediction_rms_change=float(np.sqrt(np.mean(approximate_change**2))),
        old_formula_linearized_prediction_change_quantiles=np.quantile(approximate_change,[0,.01,.5,.99,1]).tolist(),
        old_formula_mean_conditional_variance_before=float(variance_before.mean()),
        old_formula_mean_conditional_variance_after=float(variance_after.mean()),
        limitation='Gradient belongs to the older 0.60058 formula. This is a sensitivity diagnostic, not a forecast for 0.44348.')
    OUT.mkdir()
    np.savez_compressed(OUT/'posterior_repaired.npz',row_id=z['row_id'],mean=mean,covariance=cov,accepted=z['accepted'])
    report=dict(confirmation=results,diagnostic=diagnostic,
        parameters_locked_to='runs/order_prior_audit_v1/report.json',
        all_confirmation_rows_excluded_from_prior_fit=True,
        confirmation_rows_disjoint_from_first_sensor_audit=True,
        target_scores_used=False,target_rmse_forecast=None)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    (OUT/'holdout_rows.json').write_text(json.dumps(saved))
    print(json.dumps(diagnostic),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=6): main()
