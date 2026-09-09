"""Numerical convergence and sensor holdouts with frozen target coefficients."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import build_Y,map_latents as old_map,derived
from reconstruction_repair import RepairedParams,map_latents as repaired_map
from deterministic_moments import moments,normal_nodes
from frozen_current_target import predict

ROOT=Path(__file__).resolve().parents[1]


def mse_comparison(a,b,truth):
    ea=(a-truth)**2;eb=(b-truth)**2
    d=ea-eb
    return {'n':len(truth),'baseline_rmse':float(np.sqrt(ea.mean())),
        'deterministic_rmse':float(np.sqrt(eb.mean())),
        'mse_reduction':float(d.mean()),'paired_mse_reduction_se':float(d.std(ddof=1)/np.sqrt(len(d)))}


def main():
    folder=ROOT/'runs/deterministic_v1'
    path=folder/'validation.json'
    if path.exists():raise FileExistsError('Preserve validation result.')
    spec=json.loads((folder/'frozen_target.json').read_text())
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    original_cache=np.load(ROOT/'work/blocks_test.npz')
    repaired_cache=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    indices=np.random.default_rng(942).choice(len(test),1200,replace=False)
    stressful=np.argsort(repaired_cache['V_tau'])[-80:]
    selected=np.unique(np.r_[indices,stressful,np.flatnonzero(test.row_id==438963)])
    subset=test.iloc[selected]
    random=np.isin(selected,indices)
    estimates={}
    started=time.time()
    for name,params,mapper in [('original',original,old_map),('repaired',repaired,repaired_map)]:
        if name=='repaired':
            th,cov,info=mapper(build_Y(subset),params,need_cov=True,return_info=True)
            accepted=info['accepted']
        else:th,cov=mapper(build_Y(subset),params,need_cov=True);accepted=None
        for power,seed in [(8,104),(10,105),(12,107)]:
            estimates[name,power]=moments(th,cov,params,normal_nodes(power,seed),accepted)
            print(f'{name} {2**power} nodes: {time.time()-started:.1f}s',flush=True)
    predictions={power:predict(estimates['original',power],estimates['repaired',power],
        estimates['repaired',power]['pg_panel'],spec) for power in [8,10,12]}
    old_prediction=pd.read_csv(ROOT/spec['base_file']).prediction.to_numpy()[selected]
    numerical={}
    for label,mask in [('random',random),('stress',~random)]:
        reference=predictions[12][mask]
        numerical[label]={
            'mc40_vs_reference_rms':float(np.sqrt(np.mean((old_prediction[mask]-reference)**2))),
            'qmc256_vs_reference_rms':float(np.sqrt(np.mean((predictions[8][mask]-reference)**2))),
            'qmc1024_vs_reference_rms':float(np.sqrt(np.mean((predictions[10][mask]-reference)**2))),
            'n':int(mask.sum())}
    print('Numerical convergence:',json.dumps(numerical),flush=True)
    # Fresh observed-sensor rows, explicitly excluding all earlier audit samples.
    excluded=set(test[test.feature_14.notna()].sample(8000,random_state=908).row_id)
    excluded.update(test[test.feature_07.notna()].sample(5000,random_state=910).row_id)
    excluded.add(438963)
    fresh=test[~test.row_id.isin(excluded)]
    for target in [7,14,13]:
        eligible=fresh[fresh[f'feature_{target:02d}'].notna()]
        excluded.update(eligible.sample(min(6000,len(eligible)),random_state=921).row_id)
        excluded.update(eligible[(eligible.feature_04>14)|(eligible.feature_18>196)].row_id)
    for ids in json.loads((ROOT/'runs/pipeline_audit_v2/holdout_rows.json').read_text()).values():excluded.update(ids)
    fresh=test[~test.row_id.isin(excluded)]
    sensor_results=[]
    holdouts={}
    for label,target,hide in [('panel',7,[7]),('gross_frequency',14,[14,21]),('solar',12,[12])]:
        eligible=fresh[fresh[f'feature_{target:02d}'].notna()]
        subset=eligible.sample(6000,random_state=943).reset_index(drop=True)
        holdouts[label]=subset.row_id.tolist()
        truth=subset[f'feature_{target:02d}'].to_numpy()
        masked=subset.copy()
        for fid in hide:masked[f'feature_{fid:02d}']=np.nan
        th,cov,info=repaired_map(build_Y(masked),repaired,need_cov=True,return_info=True)
        # Independent MC40 draws per row match the legacy sampling method.
        from reconstruction_repair import LOW,HIGH
        rng=np.random.default_rng(944)
        L=np.linalg.cholesky(cov+np.eye(8)*1e-12)
        mc=np.zeros(len(th))
        for _ in range(40):
            ts=th+np.einsum('nij,nj->ni',L,rng.standard_normal(th.shape))
            ts[:,0]=np.clip(ts[:,0],0,1300);ts[:,3]=np.clip(ts[:,3],0,25)
            ts[:,5]=np.clip(ts[:,5],0,1.05);ts[:,6]=np.clip(ts[:,6],.005,1.05)
            ts[info['accepted']]=np.clip(ts[info['accepted']],LOW,HIGH)
            mc+=derived(ts,repaired)[{7:1,14:6,12:4}[target]]/40
        integrated=moments(th,cov,repaired,normal_nodes(8,104),info['accepted'])
        deterministic=integrated[{7:'panel',14:'E_pg',12:'solar'}[target]]
        record={'scenario':label,'target_sensor':target,**mse_comparison(mc,deterministic,truth)}
        sensor_results.append(record)
        print(json.dumps(record),flush=True)
    report={'status':'Numerical and sensor validation only; no NDEM target score.',
        'numerical':numerical,'sensor_holdouts':sensor_results,
        'frozen_target_coefficients':True,'normalization_refitted':False,
        'gate_numerical':numerical['random']['qmc256_vs_reference_rms']<numerical['random']['mc40_vs_reference_rms']/5,
        'gate_sensor_no_material_regression':all(r['deterministic_rmse']<=r['baseline_rmse']*1.005 for r in sensor_results),
        'limitations':['4096-node Sobol is an approximate numerical reference, not ground truth.',
            'These results do not prove a public or private target score improvement.',
            'Observed-sensor holdouts do not cover all condition-dependent missingness.']}
    path.write_text(json.dumps(report,indent=2))
    (folder/'holdout_rows.json').write_text(json.dumps(holdouts))
    np.savez_compressed(folder/'numerical_reference.npz',row_id=test.row_id.to_numpy()[selected],
        mc40=old_prediction,qmc256=predictions[8],qmc1024=predictions[10],qmc4096=predictions[12],random=random)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
