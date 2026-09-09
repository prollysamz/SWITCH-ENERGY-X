import numpy as np, pandas as pd, sys, pickle
sys.path.insert(0,'work')
from model3 import *
p=pickle.load(open('work/params_final.pkl','rb'))
tr=pd.read_csv('Dataset/train.csv'); sub=tr.sample(120000,random_state=99).reset_index(drop=True)
Y=build_Y(sub)
th,cov=map_latents(Y,p,n_iter=22,need_cov=True)

def blocks(th):
    rho,Tp,tau,eta,psol,pwin,pg,tl,freq=derived(th,p)
    S=th[...,6]; D=th[...,7]
    return dict(pg=pg,tau=tau,eta=eta,tl=tl,S=S,D=D,psol=psol,pwin=pwin)

# Monte-Carlo posterior expectation
rng=np.random.default_rng(0); NS=48
L=np.linalg.cholesky(cov+np.eye(NL)*1e-12)
acc={}
for s in range(NS):
    z=rng.standard_normal((len(th),NL))
    ts=th+np.einsum('nij,nj->ni',L,z)
    ts[:,3]=np.clip(ts[:,3],0,25); ts[:,6]=np.clip(ts[:,6],0.005,1.05)
    ts[:,5]=np.clip(ts[:,5],0,1.05); ts[:,0]=np.clip(ts[:,0],0,1300)
    b=blocks(ts)
    for k,v in b.items(): acc[k]=acc.get(k,0)+v/NS
B=acc
MULT={'1':np.ones_like(B['pg']), 'tau':B['tau'], 'eta':B['eta'], 'los':(1-B['tl']),
      'tau*eta':B['tau']*B['eta'], 'tau*los':B['tau']*(1-B['tl']), 'eta*los':B['eta']*(1-B['tl']),
      'tau*eta*los':B['tau']*B['eta']*(1-B['tl'])}
DEM={'0':0.0, 'D':B['D'], 'D(1-S)':B['D']*(1-B['S']), 'D-S':B['D']-B['S'],
     'S*D':B['S']*B['D'], 'D/(S+.1)':B['D']/(B['S']+0.1), 'D*(1-S)+... ':B['D']*(1-B['S'])}
print(f"{'formula':30s} {'mean':>8s} {'std':>8s} {'min':>9s} {'max':>8s}   score")
rows=[]
for mk,mv in MULT.items():
    for dk,dv in DEM.items():
        y=B['pg']*mv-dv
        mu,sd,mn,mx=y.mean(),y.std(),y.min(),y.max()
        sc=abs(mu-3.9)/0.15+abs(sd-4.6)/0.2
        rows.append((sc,f'pg*{mk} - {dk}',mu,sd,mn,mx))
for sc,name,mu,sd,mn,mx in sorted(rows):
    print(f'{name:30s} {mu:8.3f} {sd:8.3f} {mn:9.3f} {mx:8.2f}   {sc:6.2f}')
np.save('work/blocks_sub.npy', np.column_stack([B[k] for k in ['pg','tau','eta','tl','S','D','psol','pwin']]))
