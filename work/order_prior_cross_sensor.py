"""Check collateral thermal and generation effects of locked local priors."""
import json
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_order_priors import ROOT,update,compare
from model3 import build_Y,derived
from reconstruction_repair import RepairedParams
from learned_energy_distillation import stable_map

OUT=ROOT/'runs/order_prior_cross_sensor_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve run')
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    prior=np.load(ROOT/'runs/order_prior_audit_v1/test_priors.npz')
    p=RepairedParams.from_original(pickle.loads((ROOT/'work/params_final.pkl').read_bytes()),
        json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    results=[];saved={}
    for label,target,hidden in [('temperature',2,[2]),('temperature_sparse',2,[2,7,10,15]),
        ('generation',14,[14]),('generation_sparse',14,[14,13,21])]:
        available=test[(test.row_id%10==5)&test[f'feature_{target:02d}'].notna()]
        frame=available.sample(2000,random_state=99111+target).copy()
        ids=frame.index.to_numpy();truth=frame[f'feature_{target:02d}'].to_numpy().copy()
        for fid in hidden:frame[f'feature_{fid:02d}']=np.nan
        mean,cov,info=stable_map(build_Y(frame),p)
        new,newcov=update(mean,cov,prior['local'][ids],prior['sd'],p)
        base=mean[:,1] if target==2 else derived(mean,p)[6]
        candidate=new[:,1] if target==2 else derived(new,p)[6]
        record=dict(scenario=label,hidden=hidden,**compare(truth,base,candidate))
        results.append(record);saved[label]=frame.row_id.tolist()
        print(json.dumps(record),flush=True)
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(dict(results=results,target_scores_used=False,
        parameters_locked=True,all_audit_rows_excluded_from_prior_estimation=True),indent=2))
    (OUT/'holdout_rows.json').write_text(json.dumps(saved))

if __name__=='__main__':
    with threadpool_limits(limits=6):main()
