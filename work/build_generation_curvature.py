"""Test an independent squared-generation penalty with measured effects fixed.

E[generation squared] includes posterior variance. Sensor holdouts validate
reconstruction only; they do not establish that the target uses this loss term.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from build_direct_demand import ROOT, sha, residualize
from model3 import build_Y, SIG, COL
from reconstruction_repair import RepairedParams, map_latents
from deterministic_moments import moments, normal_nodes

OUT=ROOT/'runs/generation_curvature_v1'


def main():
    if OUT.exists():raise FileExistsError('Preserve experiment')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    bestpath=ROOT/ledger['best_file'];baseline=pd.read_csv(bestpath)
    test=pd.read_csv(ROOT/'Dataset/test.csv');sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if not baseline.row_id.equals(sample.row_id) or not test.row_id.equals(sample.row_id):raise ValueError('IDs differ')
    columns=[];sources=[]
    for row in ledger['results']:
        path=ROOT/row['file'];digest=sha(path)
        if 'sha256' in row and digest!=row['sha256']:raise ValueError('Scored input changed')
        f=pd.read_csv(path)
        if not f.row_id.equals(sample.row_id) or not np.isfinite(f.prediction).all():raise ValueError('Invalid source')
        columns.append(f.prediction.to_numpy());sources.append(dict(file=row['file'],sha256=digest))
    P=np.column_stack(columns);protected=[baseline.prediction.to_numpy()]
    experiments=['solar_contribution_v1','direct_demand_v1','storage_contribution_v1','compact_blend_v1']
    for experiment in experiments:
        m=json.loads((ROOT/'runs'/experiment/'manifest.json').read_text())
        a,b=ROOT/m['candidate'],ROOT/m['incumbent_file']
        if sha(a)!=m['sha256'] or sha(b)!=m['incumbent_sha256']:raise ValueError('Protected source changed')
        protected.append(pd.read_csv(a).prediction.to_numpy()-pd.read_csv(b).prediction.to_numpy())
    protected=np.column_stack(protected)
    cachepath=ROOT/'runs/deterministic_v1/moments_repaired.npz';z=np.load(cachepath)
    if not np.array_equal(z['row_id'],test.row_id):raise ValueError('Cache IDs differ')
    raw=-(z['E_pg']**2+z['V_pg'])
    residual,projection=residualize(raw,P,protected,np.ones(len(test),bool))
    stability=[];split=test.row_id.to_numpy()%2==0
    for fit in [split,~split]:
        alternate,_=residualize(raw,P,protected,fit);held=~fit
        stability.append(dict(n=int(held.sum()),correlation=float(np.corrcoef(alternate[held],residual[held])[0,1]),
            relative_rms_difference=float(np.sqrt(np.mean((alternate[held]-residual[held])**2))/residual[held].std())))
    if any(r['correlation']<.995 or r['relative_rms_difference']>.03 for r in stability):raise ValueError('Unstable direction')
    excluded=set()
    for path in (ROOT/'runs').glob('*/holdout_rows.json'):
        for ids in json.loads(path.read_text()).values():excluded.update(ids)
    available=test[test.feature_14.notna() & ~test.row_id.isin(excluded)]
    audit=available.sample(min(6000,len(available)),random_state=1061).copy()
    truth=audit.feature_14.to_numpy().copy()**2;audit['feature_14']=np.nan
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    params=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    th,cov,info=map_latents(build_Y(audit),params,need_cov=True,return_info=True)
    if info['failed_fallback'].any():raise ValueError('Failed holdout reconstruction')
    mz=moments(th,cov,params,normal_nodes(8,104),info['accepted'])
    predicted=mz['E_pg']**2+mz['V_pg']+SIG[COL[14]]**2
    # Measurement-noise variance is added only to predict the squared observed
    # sensor in this audit. Candidate raw features remain latent second moments.
    error=float(np.sqrt(np.mean((predicted-truth)**2)));constant=float(truth.std())
    if error>.4*constant:raise ValueError('Squared-generation reconstruction is too weak')
    change=.15*residual/residual.std()
    correlations=[float(np.corrcoef(change,protected[:,i])[0,1]) for i in range(protected.shape[1])]
    if max(abs(c) for c in correlations)>1e-8:raise ValueError('Protected directions changed')
    result=sample.copy();result['prediction']=baseline.prediction.to_numpy()+change
    if len(result)!=100000 or not result.row_id.is_unique or not np.isfinite(result.prediction).all():raise ValueError('Invalid candidate')
    OUT.mkdir();path=OUT/'submission_generation_curvature.csv';result.to_csv(path,index=False)
    check=pd.read_csv(path)
    if list(check.columns)!=['row_id','prediction'] or not check.row_id.equals(test.row_id):raise ValueError('Round-trip IDs/schema differ')
    np.testing.assert_allclose(check.prediction,result.prediction,atol=1e-12,rtol=1e-12)
    np.save(OUT/'direction.npy',change)
    np.savez_compressed(OUT/'sensor_audit.npz',row_id=audit.row_id.to_numpy(),truth=truth,prediction=predicted)
    (OUT/'holdout_rows.json').write_text(json.dumps(dict(squared_generation=audit.row_id.tolist())))
    report=dict(status='Unscored squared-generation loss hypothesis.',candidate=path.relative_to(ROOT).as_posix(),sha256=sha(path),
        incumbent_file=bestpath.relative_to(ROOT).as_posix(),incumbent_sha256=sha(bestpath),incumbent_rmse=ledger['best_public_rmse'],
        allow_score_calibration=True,raw_effect='-E[gross_generation squared]',effect_sd=.15,
        hypothesis='Losses may increase nonlinearly with generation; the current target formula lacks an independent squared-generation term.',
        raw_effect_sd=float(raw.std()),residual_sd=float(residual.std()),raw_penalty_increment=float(.15/residual.std()),
        independent_variance_fraction=float(residual.var()/raw.var()),stability=stability,projection=projection,
        protected_directions=['incumbent']+experiments,protected_correlations=correlations,
        sensor_audit=dict(rows=len(audit),squared_generation_rmse=error,constant_baseline_rmse=constant,r_squared=float(1-error**2/constant**2)),
        rms_change=float(np.sqrt(np.mean(change**2))),maximum_absolute_change=float(np.max(np.abs(change))),
        q_random_public_subset_se=float(np.sqrt(.7*np.var(change**2,ddof=1)/30000)),
        new_coefficient_selected_using_scores=False,target_rmse_forecast=None,sources=sources,
        cache_sha256=sha(cachepath),code_sha256={Path(__file__).name:sha(Path(__file__)),'build_direct_demand.py':sha(ROOT/'work/build_direct_demand.py')},
        limitations=['This is a target-structure experiment, not a claim that the loss law has been identified.',
            'Sensor holdouts and direction stability do not validate hidden NDEM accuracy.',
            'Sign and magnitude require measured feedback; the candidate may worsen the score.',
            'Orthogonality and sensitivity calculations use full-test moments, not the unknown public subset.'])
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['projection','sources','code_sha256']},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
