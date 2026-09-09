"""Cross-validate compact physical target laws against measured submissions.

The hidden target is never read. Each fold holds out scored submissions and
selects basis complexity by predicting their reported RMSE from the others.
"""
import json,itertools
import numpy as np,pandas as pd
from scipy.optimize import least_squares
from audit_order_priors import ROOT

OUT=ROOT/'runs/formula_recovery_cv_v1'

def main():
 if OUT.exists():raise FileExistsError('Preserve run')
 led=json.loads((ROOT/'candidates/leaderboard_results.json').read_text());sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
 rows=[r for r in led['results'] if r.get('precision')=='exact'];files=[r['file'] for r in rows];P=np.column_stack([pd.read_csv(ROOT/f).prediction.to_numpy() for f in files]);scores=np.array([r['rmse'] for r in rows]);n=P.shape[0];m_sub=P.shape[1]
 z=np.load(ROOT/'runs/order_prior_target_v1/moments_repaired.npz');sp=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text());T=(1-z['E_tau'])/sp['beta']+sp['tref'];
 v={'gen':z['E_pg'],'solar':z['solar'],'pm':z['E_pm'],'demand':z['E_D'],'storage':z['E_S'],'temp':T,'thermal':1-z['E_tau'],'trans':z['E_tl'],'panelgen':z['pg_panel']}
 raw={};raw.update(v)
 for k in list(v):raw[k+'2']=v[k]**2
 for a,b in itertools.combinations(['gen','solar','pm','demand','storage','temp','thermal','trans','panelgen'],2):raw[a+'x'+b]=v[a]*v[b]
 names=list(raw);A=np.column_stack([raw[k] for k in names]);mu=A.mean(0);sd=A.std(0);sd[sd<1e-9]=1;A=(A-mu)/sd
 candidates={
  'core':['gen','solar','pm','demand','storage','temp','thermal','trans'],
  'energy':['gen','solar','pm','demand','storage','temp','thermal','trans','panelgen','gen2','pm2','demand2','genxtemp','genxthermal','pmxdemand','demandxstorage'],
  'physics':['gen','solar','pm','demand','storage','temp','thermal','trans','panelgen','gen2','solar2','pm2','demand2','storage2','genxtemp','genxsolar','genxdemand','genxstorage','genxthermal','genxtrans','solarxdemand','solarxstorage','solarxtemp','solarxthermal','solarxtrans','pmxdemand','pmxstorage','pmxtemp','pmxthermal','pmxtrans','demandxstorage','demandxtemp','demandxthermal','demandxtrans','storagextemp','storagexthermal','storagextrans','tempxthermal','tempxtrans','thermalxtrans']}
 m=3.86235
 def fit_predict(ix,hold,ridge):
  X=A[:,ix];H=X.T@X/n;C=X.T@(P-m)/n;p2=np.mean((P-m)**2,0);d=len(ix)
  def pred(t):return np.sqrt(np.maximum(p2-2*t[:-1]@C+t[:-1]@H@t[:-1]+t[-1],1e-12))
  def fun(t):return np.r_[pred(t)[~hold]-scores[~hold],np.sqrt(ridge)*t[:-1]]
  def jac(t):
   q=pred(t);J=np.column_stack([((t[:-1]@H)[:,None]-C).T/q[:,None],.5/q]);return np.vstack([J[~hold],np.column_stack([np.sqrt(ridge)*np.eye(d),np.zeros(d)])])
  init=np.r_[np.zeros(d),.15];lo=np.r_[np.full(d,-6),0];hi=np.r_[np.full(d,6),3]
  o=least_squares(fun,init,jac=jac,bounds=(lo,hi),max_nfev=250,ftol=1e-8,xtol=1e-8,gtol=1e-8)
  return o,pred(o.x)
 hold_masks=[]
 for pats in [['physics_basis'],['order_prior'],['global_bias'],['measurement_probe'],['optimal_step']]:hold_masks.append(np.array([any(p in f for p in pats) for f in files]))
 results=[]
 for model,ks in candidates.items():
  ix=[names.index(k) for k in ks]
  for ridge in [.001,.01,.1,1.,10.]:
   errors=[]
   for hold in hold_masks:
    if not hold.any() or (~hold).sum()<3:continue
    o,implied=fit_predict(ix,hold,ridge);errors.extend((implied[hold]-scores[hold]).tolist())
    o,implied=fit_predict(ix,np.zeros(m_sub,bool),ridge);w=o.x[:-1];candidate=3.86235+A[:,ix]@w
   rec=dict(model=model,ridge=ridge,basis=ks,best_noise_rmse=float(np.sqrt(o.x[-1])),cv_rmse=float(np.sqrt(np.mean(np.array(errors)**2))),cv_errors=errors,
    all_fit_rmse=float(np.sqrt(np.mean((implied-scores)**2))),candidate_mean=float(candidate.mean()),candidate_sd=float(candidate.std()),candidate_min=float(candidate.min()),candidate_max=float(candidate.max()),rms_from_best=float(np.sqrt(np.mean((candidate-P[:,-1])**2))))
   results.append(rec);print(json.dumps(rec),flush=True)
 results.sort(key=lambda r:r['cv_rmse']);OUT.mkdir();(OUT/'report.json').write_text(json.dumps(dict(results=results,heldout_files=files,target_mean=m,limitation='Score-equation CV is only aggregate validation; no hidden labels or private score.'),indent=2))
 print('TOP',json.dumps(results[:10],indent=2),flush=True)
if __name__=='__main__':main()
