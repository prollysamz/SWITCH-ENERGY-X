"""Rich-to-masked consistency of fixed target algebra, not NDEM validation."""
import json
import pickle
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_order_priors import ROOT,update,compare
from model3 import build_Y,map_latents
from reconstruction_repair import RepairedParams
from learned_energy_distillation import stable_map
from deterministic_moments import moments,normal_nodes
from build_order_prior_candidate import design

OUT=ROOT/'runs/order_target_consistency_v1'
RICH='--rich' in sys.argv
if RICH:OUT=ROOT/'runs/order_target_consistency_rich_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve run')
    frame=pd.read_csv(ROOT/'Dataset/test.csv')
    prior=np.load(ROOT/'runs/order_prior_audit_v1/test_priors.npz')
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    spec=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    coef=np.load(ROOT/'runs/order_prior_candidate_v1/frozen_algebra.npz')['coefficients']
    nodes=normal_nodes(8,104)
    def predict(data,apply_prior):
        caches=[]
        for p in [original,repaired]:
            if p is original:
                mean,cov=map_latents(build_Y(data),p,need_cov=True);accepted=np.zeros(len(data),bool)
            else:
                mean,cov,info=stable_map(build_Y(data),p);accepted=info['accepted']
            if apply_prior:
                mean,cov=update(mean,cov,prior['local'][data.index],prior['sd'],p)
            caches.append(moments(mean,cov,p,nodes,accepted))
        a,b=caches;zero=np.zeros(len(data))
        return design(a,b,b['pg_panel'],a,b,data.feature_07.to_numpy(),spec,zero,zero)@coef
    records=[];saved={}
    for label,hidden in [('thermal',[2,7,10,15]),('storage',[8,11,19]),('demand',[9,19,21])]:
        available=frame[(frame.row_id%10==5)&frame[f'feature_{hidden[0]:02d}'].notna()]
        if RICH:
            available=available.dropna(subset=[f'feature_{k:02d}' for k in [2,7,8,9,14]])
        data=available.sample(min(1000,len(available)),random_state=78121+hidden[0]).copy()
        teacher=predict(data,False)
        for fid in hidden:data[f'feature_{fid:02d}']=np.nan
        base=predict(data,False);candidate=predict(data,True)
        record=dict(scenario=label,**compare(teacher,base,candidate))
        records.append(record);saved[label]=data.row_id.tolist()
        print(json.dumps(record),flush=True)
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(dict(results=records,
        target_scores_used=False,target_weights_fixed=True,rich_teacher=RICH,
        limitation='Teacher is the frozen formula on richer sensor observations, not true NDEM. This tests reconstruction consistency only.'),indent=2))
    (OUT/'holdout_rows.json').write_text(json.dumps(saved))

if __name__=='__main__':
    with threadpool_limits(limits=6):main()
