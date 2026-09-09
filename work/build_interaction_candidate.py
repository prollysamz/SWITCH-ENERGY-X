"""Repaired joint component reconstruction and one new thermal interaction.

This candidate tests an additional generation-dependent thermal loss. It is not
selected by a fabricated local NDEM score. Existing predictions are preserved.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import NL,build_Y,derived
from reconstruction_repair import RepairedParams,map_latents,LOW,HIGH

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orthogonal_effect(raw,predictions):
    """Remove the intercept and every already represented scored direction."""
    centered=predictions-predictions.mean(axis=0)
    u,s,_=np.linalg.svd(centered,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10))
    basis=u[:,:rank]
    mean=float(raw.mean())
    residual=raw-mean-basis@(basis.T@(raw-mean))
    sd=float(residual.std())
    if sd<1e-8:
        raise ValueError('The proposed interaction contains no new information.')
    return residual/sd,{'mean':mean,'residual_sd':sd,'existing_prediction_rank':rank,
                        'singular_values':s.tolist()}


def main():
    output=ROOT/'runs/interaction_v1'
    if output.exists():
        raise FileExistsError('Do not overwrite an existing experiment.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if not test.row_id.equals(sample.row_id):
        raise ValueError('Test and sample IDs differ.')
    incumbent=pd.read_csv(ROOT/ledger['best_file'])
    if not incumbent.row_id.equals(sample.row_id):
        raise ValueError('Best submission row IDs differ.')
    scored=[]
    sources=[]
    for item in ledger['results']:
        path=ROOT/item['file']
        digest=sha(path)
        if 'sha256' in item and digest!=item['sha256']:
            raise ValueError(f'Scored file changed: {path}')
        frame=pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id) or not np.isfinite(frame.prediction).all():
            raise ValueError(f'Invalid scored source: {path}')
        scored.append(frame.prediction.to_numpy())
        sources.append({'file':item['file'],'sha256':digest})
    output.mkdir(parents=True)
    params=RepairedParams.from_original(pickle.loads((ROOT/'work/params_final.pkl').read_bytes()),
        json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    names=['solar','wind','pg','panel','pg_panel','solar_panel','wind_panel','usable_panel',
           'usable','usable_hot40','pg_soc','pg_demand']
    features={name:np.zeros(len(test)) for name in names}
    rng=np.random.default_rng(2024)
    start=time.time()
    for a in range(0,len(test),25000):
        b=min(a+25000,len(test))
        th,cov,info=map_latents(build_Y(test.iloc[a:b]),params,need_cov=True,return_info=True)
        L=np.linalg.cholesky(cov+np.eye(NL)*1e-12)
        for _ in range(40):
            ts=th+np.einsum('nij,nj->ni',L,rng.standard_normal((b-a,NL)))
            ts[:,0]=np.clip(ts[:,0],0,1300);ts[:,3]=np.clip(ts[:,3],0,25)
            ts[:,5]=np.clip(ts[:,5],0,1.05);ts[:,6]=np.clip(ts[:,6],.005,1.05)
            ts[info['accepted']]=np.clip(ts[info['accepted']],LOW,HIGH)
            rho,panel,tau,eta,solar,wind,pg,tl,freq=derived(ts,params)
            usable=pg*tau*eta*(1-tl)
            values={'solar':solar,'wind':wind,'pg':pg,'panel':panel,'pg_panel':pg*(panel-25),
                'solar_panel':solar*(panel-25),'wind_panel':wind*(panel-25),
                'usable_panel':usable*(panel-25),'usable':usable,
                'usable_hot40':usable*np.maximum(panel-40,0),
                'pg_soc':pg*ts[:,6],'pg_demand':pg*ts[:,7]}
            for key,value in values.items(): features[key][a:b]+=value/40
        print(f'Rebuilt joint features {b}/{len(test)} in {time.time()-start:.1f}s',flush=True)
    existing=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    if not np.array_equal(existing['row_id'],test.row_id):
        raise ValueError('Repaired cache IDs differ.')
    if not np.allclose(features['pg'],existing['E_pg'],rtol=1e-10,atol=1e-10):
        raise ValueError('Joint feature reconstruction does not reproduce the repaired generation cache.')
    if not np.allclose(features['solar']+features['wind'],features['pg'],rtol=1e-12,atol=1e-12):
        raise ValueError('Solar/wind components do not reconcile.')
    np.savez_compressed(output/'joint_features.npz',row_id=test.row_id.to_numpy(),**features)
    # The negative direction asks whether thermal loss should increase with
    # generation, beyond the existing additive heat and derating terms.
    raw=-features['pg_panel']
    effect,projection=orthogonal_effect(raw,np.column_stack(scored))
    effect_sd=.5
    delta=effect_sd*effect
    result=sample.copy()
    result['prediction']=incumbent.prediction.to_numpy()+delta
    candidate=output/'submission_generation_heat.csv'
    result.to_csv(candidate,index=False)
    reread=pd.read_csv(candidate)
    if not reread.row_id.equals(sample.row_id) or len(reread)!=100000 or not np.isfinite(reread.prediction).all():
        raise ValueError('Submission validation failed.')
    np.save(output/'direction.npy',delta)
    centered=np.column_stack(scored)-np.mean(np.column_stack(scored),axis=0)
    max_cov=float(np.max(abs(centered.T@effect/len(effect))))
    report={'status':'UNSCORED new model effect; no target RMSE forecast',
        'incumbent_file':ledger['best_file'],'incumbent_rmse':ledger['best_public_rmse'],
        'incumbent_sha256':sha(ROOT/ledger['best_file']),
        'hypothesis':'Additional thermal loss depends on generation as well as temperature',
        'raw_effect':'-E[gross_generation * (panel_temperature - 25)]',
        'formula':'incumbent + 0.5 * standardized residual effect',
        'effect_sd':effect_sd,'projection':projection,'max_covariance_with_existing_predictions':max_cov,
        'sources':sources,'score_values_used_for_new_coefficient':False,
        'reconstruction':{'model':'reconstruction_repair.py','seed':2024,'draws':40,
                          'joint_not_product_of_means':True},
        'component_cache_sha256':sha(output/'joint_features.npz'),
        'candidate':str(candidate.relative_to(ROOT)),'sha256':sha(candidate),
        'prediction_mean':float(result.prediction.mean()),'prediction_sd':float(result.prediction.std(ddof=0)),
        'prediction_min':float(result.prediction.min()),'prediction_max':float(result.prediction.max()),
        'rms_change':float(np.sqrt(np.mean(delta**2))),
        'limitations':['Sign and strength of the extra thermal interaction are not identified without a score.',
                       'Projection uses unlabeled full-test moments; public moments can differ.',
                       'The candidate may score worse; keep the incumbent until measured.']}
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2),flush=True)


if __name__=='__main__':
    main()
