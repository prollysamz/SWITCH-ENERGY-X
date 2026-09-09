"""Recover fixed algebra of incumbent and isolate the deterministic branch update.

Fit is to existing predictions, never hidden labels. All legacy branches and
learned corrections stay fixed; therefore this does not claim a full refit.
"""
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_order_priors import ROOT, update
from deterministic_moments import moments, normal_nodes
from frozen_current_target import predict
from reconstruction_repair import RepairedParams
from build_hypotheses import expected_drawdown

OUT=ROOT/'runs/order_prior_target_v1'

def features(old,z,spec):
    panel=(1-z['E_tau'])/spec['beta']+spec['tref']
    return np.column_stack([predict(old,z,z['pg_panel'],spec),z['solar'],
        expected_drawdown(z),(panel-25)**2+z['V_tau']/spec['beta']**2,
        z['pg_panel'],panel,z['E_pg'],z['E_D'],z['E_S'],z['E_pg']**2+z['V_pg']])

def main():
    if OUT.exists(): raise FileExistsError('Preserve run')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    def read(path):
        df=pd.read_csv(ROOT/path)
        assert df.row_id.equals(sample.row_id)
        return df.prediction.to_numpy()
    target=read('candidates/submission_best_0p44348.csv')
    sources=[r['file'] for r in ledger['results'][:13]]
    static=[np.ones(len(target))]+[read(f) for f in sources]
    static += [read('runs/learned_energy_v1/submission_learned_energy.csv')-
        read('runs/deterministic_v1/submission_deterministic.csv')]
    static += [read('runs/missingness_correction_v1/submission_missingness_correction.csv')-
        read('runs/successful_components_v3/submission_successful_components_v3.csv')]
    raw=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    static += [raw['E_pm']-raw['E_D']]
    static=np.column_stack(static)
    spec=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    old=np.load(ROOT/'runs/deterministic_v1/moments_original.npz')
    z=np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    X=np.column_stack([static,features(old,z,spec)])
    fit=sample.row_id.to_numpy()%5!=0
    scale=X[fit].std(0);scale[scale<1e-12]=1
    coef=np.linalg.lstsq(X[fit]/scale,target[fit],rcond=1e-12)[0]/scale
    error=X@coef-target
    print('Exact algebra error:',np.max(abs(error)),flush=True)
    if np.max(abs(error))>1e-8: raise ValueError('Incomplete algebra; no candidate')
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    prior=np.load(ROOT/'runs/order_prior_audit_v1/test_priors.npz')
    caches=[]
    for name,p in [('original',original),('repaired',repaired)]:
        post=np.load(ROOT/f'runs/deterministic_v1/posterior_{name}.npz')
        mean,cov=update(post['mean'],post['covariance'],prior['local'],prior['sd'],p)
        chunks=[]
        for start in range(0,len(mean),5000):
            stop=start+5000
            chunks.append(moments(mean[start:stop],cov[start:stop],p,normal_nodes(8,104),post['accepted'][start:stop]))
        caches.append({key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]})
        print('Updated moments',name,flush=True)
    fresh=features(*caches,spec)
    delta=(fresh-X[:,-10:])@coef[-10:]
    report=dict(status='Exact incumbent algebra with deterministic reconstruction branch updated; unscored',
        max_reproduction_error=float(abs(error).max()),
        unused_partition_max_reproduction_error=float(abs(error[~fit]).max()),
        dynamic_names=['deterministic_frozen_prediction','solar','demand_storage_ratio','symmetric_heat',
            'generation_temperature','panel_temperature','generation','demand','storage','generation_second_moment'],
        dynamic_coefficients=coef[-10:].tolist(),
        rms_change=float(np.sqrt(np.mean(delta**2))),max_change=float(abs(delta).max()),
        delta_quantiles=np.quantile(delta,[0,.001,.01,.5,.99,.999,1]).tolist(),
        target_rmse_forecast=None,target_weights_retuned=False,
        limitations=['Legacy Monte Carlo predictions and learned correction remain fixed.',
                    'Laplace prior update approximates nonlinear posterior.',
                    'Sensor validation does not validate NDEM; public score is unknown.'])
    OUT.mkdir()
    np.savez_compressed(OUT/'diagnostic.npz',row_id=sample.row_id,delta=delta,coefficients=coef)
    for name,cache in zip(['original','repaired'],caches):
        np.savez_compressed(OUT/f'moments_{name}.npz',row_id=sample.row_id,**cache)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=6): main()
