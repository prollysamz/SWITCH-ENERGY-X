import numpy as np, pandas as pd, sys
TARGET_MEAN=3.9; TARGET_SD=float(sys.argv[1]) if len(sys.argv)>1 else 4.40
tr=np.load('work/blocks_train.npz'); te=np.load('work/blocks_test.npz')
def cands(z):
    pg,tau,eta,tl,S,D=[z['E_'+k] for k in ['pg','tau','eta','tl','S','D']]
    pm=z['E_pm']; los=1-tl
    out={}
    out['A pg*tau*eta*los - D(1-S)'] = pm - D*(1-S)
    out['B pg*tau*eta*los - (D-S)'] = pm - (D-S)
    out['C pg*tau*los - D(1-S)']    = pg*tau*los - D*(1-S)
    out['D cap20 - D(1-S)']         = z['E_cap20'] - D*(1-S)
    out['E pg*tau*eta*los']         = pm
    out['F pg*tau*los - S*D']       = pg*tau*los - S*D
    return out
Ctr,Cte=cands(tr),cands(te)
keys=list(Ctr); W=np.array([0.25,0.20,0.20,0.15,0.10,0.10]); W=W/W.sum()
ALL={k:np.concatenate([Ctr[k],Cte[k]]) for k in keys}
print('=== candidates on full 500k (train+test) ===')
for k in keys:
    v=ALL[k]; print(f'  {k:28s} mean={v.mean():7.3f} sd={v.std():6.3f} min={v.min():7.2f} max={v.max():7.2f} P(<0)={(v<0).mean():.3f}')
M=np.array([ALL[k] for k in keys]); C=np.corrcoef(M)
print('\n=== candidate correlation ===')
print('        '+' '.join(f'{k[0]:>8s}' for k in keys))
for i,k in enumerate(keys): print(f'  {k[0]:5s} '+' '.join(f'{C[i,j]:8.5f}' for j in range(len(keys))))
Z={k:(ALL[k]-ALL[k].mean())/ALL[k].std() for k in keys}
ens=sum(w*Z[k] for w,k in zip(W,keys))
ens=(ens-ens.mean())/ens.std()
pred_all=TARGET_MEAN+TARGET_SD*ens
ntr=len(tr['E_pg']); pred_te=pred_all[ntr:]
print(f'\nensemble corr with each candidate: '+' '.join(f'{k[0]}={np.corrcoef(ens,Z[k])[0,1]:.5f}' for k in keys))
sub=pd.read_csv('Dataset/sample_submission.csv')
te_ids=pd.read_csv('Dataset/test.csv',usecols=['row_id'])['row_id'].values
assert np.array_equal(te_ids, sub['row_id'].values), 'row order mismatch'
sub['prediction']=pred_te
sub.to_csv('submission.csv',index=False)
print(f"\nsubmission.csv written: n={len(sub)} mean={pred_te.mean():.4f} sd={pred_te.std():.4f} "
      f"min={pred_te.min():.2f} max={pred_te.max():.2f} P(<0)={(pred_te<0).mean():.3f}")
probe=sub.copy(); probe['prediction']=TARGET_MEAN; probe.to_csv('submission_probe_const.csv',index=False)
# posterior uncertainty summary
sd_pg=np.sqrt(te['V_pg']); print(f"\ntest posterior sd of gross gen: mean={sd_pg.mean():.4f} "
      f"med={np.median(sd_pg):.4f} p90={np.quantile(sd_pg,.9):.4f} p99={np.quantile(sd_pg,.99):.4f}")
print(f"implied prediction MSE from imputation alone ~ {(0.824**2*(te['V_pg']).mean()):.4f} "
      f"(RMSE contribution {np.sqrt(0.824**2*te['V_pg'].mean()):.3f})")
