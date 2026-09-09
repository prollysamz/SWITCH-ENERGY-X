"""Generate one locally validated numerical correction with frozen target.

No new target effect, score fit, normalization or latent model is introduced.
"""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import build_Y,map_latents as old_map
from reconstruction_repair import RepairedParams,map_latents as repaired_map
from deterministic_moments import moments,normal_nodes
from frozen_current_target import predict

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'runs/deterministic_v1'
    output=folder/'submission_deterministic.csv'
    if output.exists() or (folder/'moments_original.npz').exists():raise FileExistsError('Do not overwrite a reconstruction run.')
    spec=json.loads((folder/'frozen_target.json').read_text())
    validation=json.loads((folder/'validation.json').read_text())
    if not validation['gate_numerical'] or not validation['gate_sensor_no_material_regression']:
        raise ValueError('Required local validation gates did not pass.')
    basepath=ROOT/spec['base_file']
    if sha(basepath)!=spec['base_sha256']:raise ValueError('Best snapshot changed.')
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if not test.row_id.equals(sample.row_id):raise ValueError('Test/sample IDs differ.')
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    nodes=normal_nodes(8,104)
    estimates={}
    counts={}
    started=time.time()
    for name,params,mapper in [('original',original,old_map),('repaired',repaired,repaired_map)]:
        full={}
        all_th=[];all_cov=[];all_accepted=[]
        counts[name]={}
        for a in range(0,len(test),10000):
            b=min(a+10000,len(test))
            Y=build_Y(test.iloc[a:b])
            if name=='repaired':
                th,cov,info=mapper(Y,params,need_cov=True,return_info=True)
                accepted=info['accepted']
                for key,value in info.items():
                    if value.dtype==bool:counts[name][key]=counts[name].get(key,0)+int(value.sum())
                if info['failed_fallback'].any() or info['still_inconsistent'].any():
                    raise ValueError('Unresolved reconstruction; inspect before producing candidate.')
            else:th,cov=mapper(Y,params,need_cov=True);accepted=None
            z=moments(th,cov,params,nodes,accepted)
            for key,value in z.items():
                if key not in full:full[key]=np.empty(len(test))
                full[key][a:b]=value
            all_th.append(th);all_cov.append(cov)
            all_accepted.append(accepted if accepted is not None else np.zeros(len(th),bool))
            print(f'{name} {b}/{len(test)}: {time.time()-started:.1f}s',flush=True)
        full['row_id']=test.row_id.to_numpy()
        estimates[name]=full
        np.savez_compressed(folder/f'moments_{name}.npz',**full)
        np.savez_compressed(folder/f'posterior_{name}.npz',row_id=test.row_id.to_numpy(),
            mean=np.concatenate(all_th),covariance=np.concatenate(all_cov),accepted=np.concatenate(all_accepted))
    prediction=predict(estimates['original'],estimates['repaired'],estimates['repaired']['pg_panel'],spec)
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction.')
    sample['prediction']=prediction
    sample.to_csv(output,index=False)
    check=pd.read_csv(output)
    if not check.row_id.equals(test.row_id) or not np.allclose(check.prediction,prediction,atol=1e-12,rtol=1e-12):
        raise ValueError('Output validation failed.')
    reference=np.load(folder/'numerical_reference.npz')
    ix=pd.Index(test.row_id).get_indexer(reference['row_id'])
    if not np.allclose(prediction[ix],reference['qmc256'],atol=1e-9,rtol=0):
        raise ValueError('Full run differs from independently batched validation.')
    old=pd.read_csv(basepath).prediction.to_numpy()
    if sha(basepath)!=spec['base_sha256']:raise ValueError('Preserved best changed.')
    report={'status':'UNSCORED numerical correction; locally sensor-validated.',
        'candidate':output.relative_to(ROOT).as_posix(),'sha256':sha(output),
        'incumbent_file':spec['base_file'],'incumbent_sha256':spec['base_sha256'],'incumbent_rmse':spec['base_rmse'],
        'target_weights_frozen':True,'normalization_refitted':False,'nodes':256,'seed':104,
        'rms_prediction_change':float(np.sqrt(np.mean((prediction-old)**2))),
        'max_prediction_change':float(abs(prediction-old).max()),
        'prediction_min':float(prediction.min()),'prediction_max':float(prediction.max()),
        'fit_counts':counts,'full_run_matches_validation':True,
        'sources':{f:sha(ROOT/f) for f in ['work/deterministic_moments.py','work/frozen_current_target.py',
            'work/reconstruction_repair.py','work/model3.py','runs/deterministic_v1/frozen_target.json',
            'runs/deterministic_v1/validation.json']},
        'limitations':['No target score has been measured; numerical accuracy is not target accuracy.',
            'Retains the legacy and repaired reconstruction branches and Gaussian posterior assumption.']}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
