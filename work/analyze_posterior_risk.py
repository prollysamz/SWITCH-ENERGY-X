"""Propagate current-model uncertainty to locate weak information regimes.

This is conditional on the inferred latent model and target formula. It is not
an identified irreducible noise floor and is not a target RMSE estimate.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import derived,PRIOR_SD
from reconstruction_repair import RepairedParams
from frozen_current_target import predict

ROOT=Path(__file__).resolve().parents[1]


def point_blocks(th,p):
    _,panel,tau,eta,_,_,pg,tl,_=derived(th,p)
    values=dict(pg=pg,tau=tau,eta=eta,tl=tl,S=th[:,6],D=th[:,7])
    z={'E_'+k:v for k,v in values.items()}
    z.update({'V_'+k:np.zeros(len(th)) for k in values})
    z['E_pm']=pg*tau*eta*(1-tl)
    for c in [16,18,20,22]:z[f'E_cap{c}']=np.minimum(c,z['E_pm'])
    return z,pg*(panel-25)


def main():
    folder=ROOT/'runs/deterministic_v1'
    output=folder/'posterior_risk.json'
    if output.exists():raise FileExistsError('Preserve diagnostic.')
    spec=json.loads((folder/'frozen_target.json').read_text())
    cache=np.load(folder/'posterior_repaired.npz')
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    if not np.array_equal(cache['row_id'],test.row_id):raise ValueError('IDs differ.')
    th,cov=cache['mean'],cache['covariance']
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    repaired=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    def f(x):
        old,_=point_blocks(x,original)
        rep,joint=point_blocks(x,repaired)
        return predict(old,rep,joint,spec)
    J=np.empty_like(th)
    for k in range(8):
        step=1e-4*PRIOR_SD[k]
        plus=th.copy();minus=th.copy()
        plus[:,k]+=step;minus[:,k]-=step
        J[:,k]=(f(plus)-f(minus))/(2*step)
    variance=np.maximum(np.einsum('ni,nij,nj->n',J,cov,J),0)
    have=lambda k:test[f'feature_{k:02d}'].notna().to_numpy()
    regimes={
        'all':np.ones(len(th),bool),
        'panel_or_loss_observed':have(7)|have(15),
        'panel_and_loss_missing':~have(7)&~have(15),
        'panel_loss_and_ambient_missing':~have(7)&~have(15)&~have(2),
        'gross_or_frequency_observed':have(14)|have(21),
        'gross_and_frequency_missing':~have(14)&~have(21),
        'all_direct_wind_indicators_missing':~have(4)&~have(13)&~have(14)&~have(18)&~have(21)}
    results=[]
    for label,mask in regimes.items():
        if not mask.any():continue
        results.append({'regime':label,'n':int(mask.sum()),'fraction':float(mask.mean()),
            'conditional_uncertainty_rms':float(np.sqrt(variance[mask].mean())),
            'share_of_total_linearized_variance':float(variance[mask].sum()/variance.sum())})
    report={'status':'Conditional linearized uncertainty diagnostic; not target validation.',
        'results':results,'top_1pct_variance_share':float(np.sort(variance)[-len(th)//100:].sum()/variance.sum()),
        'limitations':['Gaussian posterior and assumed target formula may be wrong.',
            'Local linearization can miss nonlinear and multimodal uncertainty.',
            'The two model branches are evaluated around the repaired state for this diagnostic.',
            'Regimes overlap and their variance shares must not be added.']}
    output.write_text(json.dumps(report,indent=2))
    np.savez_compressed(folder/'posterior_risk.npz',row_id=test.row_id.to_numpy(),variance=variance,gradient=J)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
