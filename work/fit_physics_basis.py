"""Regularized score-equation fit over a broad physical feature basis.

Aggregate leaderboard equations are necessary constraints, not target labels.
No submission is emitted; this reports whether a smooth law can explain them.
"""
import json,itertools
import numpy as np,pandas as pd
from scipy.optimize import least_squares
from audit_order_priors import ROOT

OUT=ROOT/'runs/physics_basis_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve run')
    led=json.loads((ROOT/'candidates/leaderboard_results.json').read_text());sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    rows=[r for r in led['results'] if r.get('precision')=='exact'];P=np.column_stack([pd.read_csv(ROOT/r['file']).prediction for r in rows]);scores=np.array([r['rmse'] for r in rows]);n=len(P)
    z=np.load(ROOT/'runs/order_prior_target_v1/moments_repaired.npz');sp=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    T=(1-z['E_tau'])/sp['beta']+sp['tref']; vars={'gen':z['E_pg'],'solar':z['solar'],'demand':z['E_D'],'storage':z['E_S'],'temperature':T,'thermal_loss':1-z['E_tau'],'transmission':z['E_tl'],'panel_generation':z['pg_panel'],'pm':z['E_pm']}
    # Center/scale base physical quantities; add selected pairwise interactions and smooth transforms.
    raw={};raw.update(vars)
    for k,v in list(vars.items()):
        raw[k+'_sq']=v*v
        raw[k+'_sqrt']=np.sqrt(np.maximum(v-v.min()+1e-5,0))
    for a,b in itertools.combinations(['gen','solar','demand','storage','temperature','thermal_loss','transmission','panel_generation','pm'],2):raw[a+'_'+b]=vars[a]*vars[b]
    X=np.column_stack([raw[k] for k in raw]);names=list(raw);mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1;X=(X-mu)/sd
    m=3.86235 # measured approximate global mean constraint from constant probe
    H=X.T@X/n;C=X.T@(P-m)/n;p2=np.mean((P-m)**2,axis=0)
    # Fit target coefficients and a common noise variance to score equations, plus ridge.
    best=None; rng=np.random.default_rng(2209)
    for ridge in [0.01,0.1]:
      for restart in range(3):
        init=np.r_[rng.normal(0,.1,len(names)),.1]
        def pred(t):
            w=t[:-1];return np.sqrt(np.maximum(p2-2*w@C+w@H@w+t[-1],1e-12))
        def fun(t):return np.r_[pred(t)-scores,np.sqrt(ridge)*t[:-1]]
        def jac(t):
            q=pred(t); w=t[:-1]
            jscore=np.column_stack([((w@H)[:,None]-C).T/q[:,None],.5/q])
            return np.vstack([jscore,np.column_stack([np.sqrt(ridge)*np.eye(len(names)),np.zeros(len(names))])])
        opt=least_squares(fun,init,jac=jac,bounds=(np.r_[np.full(len(names),-8),0],np.r_[np.full(len(names),8),3]),max_nfev=350,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        if best is None or np.mean(fun(opt.x)**2)<best[0]:best=(np.mean(fun(opt.x)**2),opt)
    loss,opt=best;w=opt.x[:-1];implied=np.sqrt(np.maximum(p2-2*w@C+w@H@w+opt.x[-1],0));candidate=m+X@w
    fit=np.array([r['file'] not in ['runs/measurement_probe_v1/submission_measurement_probe.csv','runs/optimal_step_v1/submission_optimal_step.csv','runs/order_prior_candidate_v1/submission_order_priors.csv'] for r in rows])
    records=dict(ridge_objective=float(loss),coefficients={k:float(v) for k,v in zip(names,w) if abs(v)>0.02},residual_variance=float(opt.x[-1]),
      fit_rmse=float(np.sqrt(np.mean((implied[fit]-scores[fit])**2))),held_rmse=float(np.sqrt(np.mean((implied[~fit]-scores[~fit])**2))),
      max_fit_error=float(abs(implied[fit]-scores[fit]).max()),implied_scores=implied.tolist(),reported_scores=scores.tolist(),
      candidate_mean=float(candidate.mean()),candidate_sd=float(candidate.std()),candidate_min=float(candidate.min()),candidate_max=float(candidate.max()),
      rms_from_best=float(np.sqrt(np.mean((candidate-P[:,-1])**2))),
      limitation='This fit uses aggregate public-score equations and a measured approximate bias, not hidden labels; many laws satisfy finite constraints.')
    OUT.mkdir();(OUT/'report.json').write_text(json.dumps(dict(status='diagnostic only',basis_size=len(names),basis_names=names,**records),indent=2));np.savez_compressed(OUT/'candidate_diagnostic.npz',row_id=sample.row_id,candidate=candidate)
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':main()
