"""Honest validation: hide observed features, see how well the model recovers them."""
import numpy as np, pandas as pd, sys, pickle
sys.path.insert(0,'work')
from model3 import *
p=pickle.load(open('work/params_final.pkl','rb'))
tr=pd.read_csv('Dataset/train.csv').sample(30000,random_state=5).reset_index(drop=True)
Y0=build_Y(tr)
def pgpred(Y):
    th,_=map_latents(Y,p,n_iter=22)
    rho,Tp,tau,eta,psol,pwin,pg,tl,freq=derived(th,p)
    return pg, pg*tau*eta*(1-tl)
pg_full,_=pgpred(Y0)
print('Recovery of GROSS GENERATION (feature_14) when hidden, by what else survives:\n')
scen=[('f14 hidden',[14]),
      ('f14+f21 hidden',[14,21]),
      ('f14+f21+f13 hidden',[14,21,13]),
      ('f14+f21+f13+f12 hidden',[14,21,13,12]),
      ('f14+f21+f13+f12+f18 hidden',[14,21,13,12,18]),
      ('above +f04 hidden (no wind info)',[14,21,13,12,18,4])]
have=~np.isnan(Y0[:,COL[14]])
truth=Y0[have,COL[14]]
for name,hide in scen:
    Y=Y0.copy()
    for f in hide: Y[:,COL[f]]=np.nan
    pg,_=pgpred(Y)
    e=pg[have]-truth
    print(f'  {name:36s} n={have.sum():6d}  RMSE={np.sqrt((e**2).mean()):.4f}  MAE={np.abs(e).mean():.4f}  bias={e.mean():+.4f}')
print('\n(feature_14 itself carries observation noise sigma~0.032, so that is the floor)')
# what fraction of TEST rows falls in each information regime?
te=pd.read_csv('Dataset/test.csv')
o={i:te[f'feature_{i:02d}'].notna().values for i in [12,13,14,18,21,4]}
n=len(te)
r1=o[14]; r2=~o[14]&o[21]; r3=~o[14]&~o[21]&(o[13]&o[12]); r4=~o[14]&~o[21]&~(o[13]&o[12])
print(f'\nTEST information regimes: f14 observed {r1.mean():.4f} | no f14 but f21 {r2.mean():.4f} | '
      f'f12+f13 {r3.mean():.4f} | weakest {r4.mean():.4f}')
