"""Produce a versioned fixed-target reconstruction candidate after sensor QA.

Legacy caches, model parameters and scored submissions are never overwritten.
The original 40 samples, seed, chunking and target normalization are preserved.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import NL, build_Y, derived
from reconstruction_repair import RepairedParams,map_latents,LOW,HIGH
from frozen_target import freeze,predict

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'runs/reconstruction_v1'
    if folder.exists():
        raise FileExistsError('Versioned output exists; it must not be overwritten.')
    validation=json.loads((ROOT/'work/reconstruction_repair_validation.json').read_text())['results']
    old={(r['scenario'],r['regime']):r for r in validation if r['method']=='original'}
    for r in validation:
        if r['method']=='repair' and r['regime'] in ['all','ordinary_wind']:
            base=old[(r['scenario'],r['regime'])]['rmse']
            if r['rmse']>base*1.01+1e-5:
                raise RuntimeError('Sensor validation regression exceeds the stated tolerance.')
    protected=['Dataset/train.csv','Dataset/test.csv','Dataset/sample_submission.csv',
               'work/blocks_train.npz','work/blocks_test.npz','work/params_final.pkl',
               'candidates/submission_best_0p78886.csv','candidates/manifest.json',
               'candidates/refined_plane_manifest.json']
    before={p:sha(ROOT/p) for p in protected}
    original=pickle.load(open(ROOT/'work/params_final.pkl','rb'))
    params=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    df=pd.read_csv(ROOT/'Dataset/test.csv')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if not df.row_id.equals(sample.row_id) or not df.row_id.is_unique:
        raise ValueError('Test IDs do not match submission IDs.')
    frozen=freeze(np.load(ROOT/'work/blocks_train.npz'),np.load(ROOT/'work/blocks_test.npz'),
                  json.loads((ROOT/'candidates/manifest.json').read_text()),
                  json.loads((ROOT/'candidates/refined_plane_manifest.json').read_text()))
    incumbent=pd.read_csv(ROOT/'candidates/submission_best_0p78886.csv').prediction.to_numpy()
    baseline_rebuilt=predict(np.load(ROOT/'work/blocks_test.npz'),frozen)
    if not np.allclose(baseline_rebuilt,incumbent,rtol=1e-12,atol=1e-12):
        raise RuntimeError('Frozen target does not reproduce the scored incumbent.')
    folder.mkdir(parents=True)
    (folder/'frozen_target.json').write_text(json.dumps(frozen,indent=2))
    names=['pg','tau','eta','tl','S','D']
    out={prefix+k:np.zeros(len(df)) for prefix in ['E_','V_'] for k in names}
    out['E_pm']=np.zeros(len(df))
    for cap in [16,18,20,22]: out[f'E_cap{cap}']=np.zeros(len(df))
    flags={key:np.zeros(len(df),dtype=bool) for key in ['retry','accepted','still_inconsistent']}
    residual=np.zeros(len(df))
    rng=np.random.default_rng(2024)
    start=time.time()
    for a in range(0,len(df),25000):
        b=min(a+25000,len(df))
        th,cov,info=map_latents(build_Y(df.iloc[a:b]),params,need_cov=True,return_info=True)
        L=np.linalg.cholesky(cov+np.eye(NL)*1e-12)
        for key in flags: flags[key][a:b]=info[key]
        residual[a:b]=info['max_residual_after']
        acc={k:np.zeros(b-a) for k in out}
        for _ in range(40):
            ts=th+np.einsum('nij,nj->ni',L,rng.standard_normal((b-a,NL)))
            # Preserve the original common-random-number sampling protocol.
            ts[:,0]=np.clip(ts[:,0],0,1300);ts[:,3]=np.clip(ts[:,3],0,25)
            ts[:,5]=np.clip(ts[:,5],0,1.05);ts[:,6]=np.clip(ts[:,6],.005,1.05)
            # Bounded repair posteriors must not sample back into invalid states.
            changed=info['accepted']
            ts[changed]=np.clip(ts[changed],LOW,HIGH)
            rho,Tp,tau,eta,solar,wind,pg,tl,freq=derived(ts,params)
            values=dict(pg=pg,tau=tau,eta=eta,tl=tl,S=ts[:,6],D=ts[:,7])
            for key,value in values.items():
                acc['E_'+key]+=value/40
                acc['V_'+key]+=value*value/40
            pm=pg*tau*eta*(1-tl)
            acc['E_pm']+=pm/40
            for cap in [16,18,20,22]: acc[f'E_cap{cap}']+=np.minimum(cap,pm)/40
        for key in names:
            acc['V_'+key]=np.maximum(acc['V_'+key]-acc['E_'+key]**2,0)
        for key,value in acc.items(): out[key][a:b]=value
        print(f'Reconstructed {b}/{len(df)}; retried {int(info["retry"].sum())}, '
              f'flagged {int(info["still_inconsistent"].sum())}; {time.time()-start:.1f}s',flush=True)
    out['row_id']=df.row_id.to_numpy()
    np.savez_compressed(folder/'blocks_test.npz',**out)
    predicted=predict(out,frozen)
    if not np.isfinite(predicted).all():
        raise RuntimeError('Nonfinite predictions.')
    sample['prediction']=predicted
    output=folder/'submission_reconstruction_repaired.csv'
    sample.to_csv(output,index=False)
    loaded=pd.read_csv(output)
    if list(loaded.columns)!=['row_id','prediction'] or len(loaded)!=100000 or not loaded.row_id.equals(df.row_id):
        raise RuntimeError('Submission schema/alignment failure.')
    if not np.allclose(loaded.prediction,predicted,rtol=1e-12,atol=1e-12):
        raise RuntimeError('CSV round-trip mismatch.')
    delta=predicted-incumbent
    diagnostics=pd.DataFrame({'row_id':df.row_id,**flags,'max_sensor_residual':residual,
                              'old_prediction':incumbent,'new_prediction':predicted,'change':delta})
    diagnostics.loc[flags['retry']|flags['still_inconsistent']|(abs(delta)>.25)].to_csv(folder/'affected_rows.csv',index=False)
    after={p:sha(ROOT/p) for p in protected}
    if before!=after: raise RuntimeError('Protected legacy artifacts changed.')
    source_files=['work/reconstruction_repair.py','work/frozen_target.py','work/run_reconstruction_repair.py',
                  'work/wind_repair_v2.json','work/reconstruction_repair_validation.json']
    report={'status':'UNSCORED reconstruction-only candidate; target and normalization frozen',
        'protected_input_hashes':before,'source_hashes':{p:sha(ROOT/p) for p in source_files},
        'output':str(output.relative_to(ROOT)),'sha256':sha(output),
        'frozen_target_sha256':sha(folder/'frozen_target.json'),'cache_sha256':sha(folder/'blocks_test.npz'),
        'n':len(df),'normalization_refitted':False,'target_weights_retuned':False,
        'monte_carlo':{'seed':2024,'samples':40,'chunk_size':25000},
        'fit_counts':{k:int(v.sum()) for k,v in flags.items()},
        'prediction_change_rms':float(np.sqrt(np.mean(delta**2))),
        'changed_over_1e_minus8':int(np.sum(abs(delta)>1e-8)),
        'changed_over_0p25':int(np.sum(abs(delta)>.25)),
        'max_absolute_change':float(np.max(abs(delta))),
        'prediction_mean':float(predicted.mean()),'prediction_sd':float(predicted.std()),
        'prediction_min':float(predicted.min()),'prediction_max':float(predicted.max()),
        'known_regression_row':diagnostics.loc[df.row_id==438963].to_dict(orient='records')[0],
        'limitations':['Sensor holdouts do not measure NDEM target RMSE.',
                       'Rare high-wind test strata contain only 10 or 11 rows.',
                       'Posterior uncertainty remains a local Gaussian approximation.',
                       'Any remaining inconsistent rows are explicitly reported.']}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if 'hash' not in k},indent=2),flush=True)


if __name__=='__main__':
    main()
