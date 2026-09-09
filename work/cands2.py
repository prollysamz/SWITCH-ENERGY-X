import numpy as np
B=np.load('work/blocks_sub.npy')
pg,tau,eta,tl,S,D,psol,pwin=[B[:,i] for i in range(8)]
los=1-tl
def stats(y):
    return y.mean(),y.std(),y.min(),y.max(),(y<0).mean(),((y-y.mean())**3).mean()/y.std()**3
MULT={'1':np.ones_like(pg),'tau':tau,'eta':eta,'los':los,'tau*eta':tau*eta,'tau*los':tau*los,
      'eta*los':eta*los,'tau*eta*los':tau*eta*los}
DEM={'0':np.zeros_like(D),'D':D,'D(1-S)':D*(1-S),'D-S':D-S,'S*D':S*D,'D/S':D/np.clip(S,1e-3,None),
     'D(1-S)/S':D*(1-S)/np.clip(S,1e-3,None),'D/(S+.25)':D/(S+0.25),'2D(1-S)':2*D*(1-S),
     '4D(1-S)':4*D*(1-S),'D+2(1-S)':D+2*(1-S),'D*(2-S)':D*(2-S)}
CAP=[None,16.0,18.0,20.0,22.0,25.0]
rows=[]
for mk,mv in MULT.items():
    for dk,dv in DEM.items():
        for c in CAP:
            g=pg*mv if c is None else np.minimum(c,pg*mv)
            y=g-dv
            mu,sd,mn,mx,pn,sk=stats(y)
            sc=abs(mu-3.9)/0.10+abs(sd-4.6)/0.15
            rows.append((sc,f'{"min(%g,"%c if c else ""}pg*{mk}{")" if c else ""} - {dk}',mu,sd,mn,mx,pn,sk))
rows.sort()
print(f"{'formula':38s} {'mean':>7s} {'std':>7s} {'min':>8s} {'max':>7s} {'P(<0)':>7s} {'skew':>6s}  score")
for sc,n,mu,sd,mn,mx,pn,sk in rows[:28]:
    print(f'{n:38s} {mu:7.3f} {sd:7.3f} {mn:8.2f} {mx:7.2f} {pn:7.3f} {sk:6.2f}  {sc:5.2f}')
print('\n--- how different are the leading candidates really? (corr after centring) ---')
top=[]
for sc,n,*_ in rows[:14]:
    pass
def build(name):
    for sc,n,mu,sd,mn,mx,pn,sk in rows:
        if n==name: pass
    return None
sel={}
for mk in ['tau*eta*los','tau*los','eta*los','los','tau*eta']:
    for dk in ['0','D','D(1-S)','D-S','S*D']:
        sel[f'{mk}|{dk}']=pg*MULT[mk]-DEM[dk]
ks=list(sel); M=np.array([sel[k] for k in ks])
Cm=np.corrcoef(M)
print('    '+' '.join(f'{k[:11]:>12s}' for k in ks[:8]))
for i,k in enumerate(ks[:8]):
    print(f'{k[:11]:12s}'+' '.join(f'{Cm[i,j]:12.5f}' for j in range(8)))
