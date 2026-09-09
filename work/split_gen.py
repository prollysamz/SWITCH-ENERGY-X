import numpy as np, pandas as pd, sys, pickle
sys.path.insert(0,'work')
from model3 import *
p=pickle.load(open('work/params_final.pkl','rb'))
te=pd.read_csv('Dataset/test.csv'); Y=build_Y(te); n=len(Y)
psol=np.zeros(n); pwin=np.zeros(n)
for a in range(0,n,25000):
    b=min(a+25000,n)
    th,_=map_latents(Y[a:b],p,n_iter=22)
    rho,Tp,tau,eta,ps,pw,pg,tl,freq=derived(th,p)
    psol[a:b]=ps; pwin[a:b]=pw
    print('  %d/%d'%(b,n),flush=True)
np.savez_compressed('work/split_test.npz',psol=psol,pwin=pwin,row_id=te.row_id.values)
print('psol mean=%.4f sd=%.4f | pwin mean=%.4f sd=%.4f'%(psol.mean(),psol.std(),pwin.mean(),pwin.std()))
