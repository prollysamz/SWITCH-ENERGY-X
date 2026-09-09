"""Check mask likelihood weighting on new observed-sensor holdouts.

Artificially hidden values are withheld from inference. Their natural observed
status is retained in the mask likelihood, since the evaluation sample was
selected by that status. This does not validate naturally missing targets.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import build_Y,derived
from reconstruction_repair import RepairedParams,map_latents,LOW,HIGH
from deterministic_moments import normal_nodes
from missingness_likelihood import failure_weights
from validate_deterministic_moments import mse_comparison

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'runs/missingness_v1'
    output=folder/'validation.json'
    if output.exists():raise FileExistsError('Preserve validation.')
    model=json.loads((folder/'model.json').read_text())
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    params=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    excluded=set(test[test.feature_14.notna()].sample(8000,random_state=908).row_id)
    excluded.update(test[test.feature_07.notna()].sample(5000,random_state=910).row_id)
    excluded.add(438963)
    fresh=test[~test.row_id.isin(excluded)]
    for target in [7,14,13]:
        eligible=fresh[fresh[f'feature_{target:02d}'].notna()]
        excluded.update(eligible.sample(min(6000,len(eligible)),random_state=921).row_id)
        excluded.update(eligible[(eligible.feature_04>14)|(eligible.feature_18>196)].row_id)
    for path in ['runs/pipeline_audit_v2/holdout_rows.json','runs/deterministic_v1/holdout_rows.json']:
        for ids in json.loads((ROOT/path).read_text()).values():excluded.update(ids)
    fresh=test[~test.row_id.isin(excluded)]
    records=[];saved_ids={}
    for label,target,hide in [('panel',7,[7]),('panel_loss',7,[7,15]),
            ('panel_loss_ambient',7,[7,15,2]),('humidity',3,[3])]:
        eligible=fresh[fresh[f'feature_{target:02d}'].notna()]
        subset=eligible.sample(min(8000,len(eligible)),random_state=947).reset_index(drop=True)
        saved_ids[label]=subset.row_id.tolist()
        truth=subset[f'feature_{target:02d}'].to_numpy()
        natural_mask=subset[[f'feature_{i:02d}' for i in range(1,26)]].isna().to_numpy(dtype=int)
        masked=subset.copy()
        for fid in hide:masked[f'feature_{fid:02d}']=np.nan
        th,cov,info=map_latents(build_Y(masked),params,need_cov=True,return_info=True)
        L=np.linalg.cholesky(cov+np.eye(8)*1e-12)
        nodes=normal_nodes(8,104)
        uniform=np.zeros(len(th));weighted=np.zeros(len(th));normalizer=np.zeros(len(th))
        for node in nodes:
            ts=th+np.einsum('nij,j->ni',L,node)
            ts[:,0]=np.clip(ts[:,0],0,1300);ts[:,3]=np.clip(ts[:,3],0,25)
            ts[:,5]=np.clip(ts[:,5],0,1.05);ts[:,6]=np.clip(ts[:,6],.005,1.05)
            ts[info['accepted']]=np.clip(ts[info['accepted']],LOW,HIGH)
            value=derived(ts,params)[1] if target==7 else ts[:,2]
            weight=failure_weights(ts,natural_mask,params,model)
            uniform+=value/len(nodes)
            weighted+=weight*value
            normalizer+=weight
        weighted/=normalizer
        record={'scenario':label,'target_sensor':target,**mse_comparison(uniform,weighted,truth),
            'rms_prediction_change':float(np.sqrt(np.mean((uniform-weighted)**2)))}
        records.append(record)
        print(json.dumps(record),flush=True)
    credible=[r for r in records if r['mse_reduction']>3*r['paired_mse_reduction_se']]
    regression=[r for r in records if r['mse_reduction']< -2*r['paired_mse_reduction_se']]
    report={'status':'Held-out observed-sensor evaluation; no NDEM score.',
        'results':records,'credible_improvement_scenarios':[r['scenario'] for r in credible],
        'regression_scenarios':[r['scenario'] for r in regression],
        'gate_for_full_candidate':bool(credible) and not regression,
        'limitations':['Natural mask retained for artificially hidden sensors to account for observed-sample selection.',
            'No ground truth exists for naturally missing cells; transfer to those cells is unverified.',
            'Fitted failure probabilities and Gaussian posterior approximation can be imperfect.']}
    output.write_text(json.dumps(report,indent=2))
    (folder/'holdout_rows.json').write_text(json.dumps(saved_ids))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
