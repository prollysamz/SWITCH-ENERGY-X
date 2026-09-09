import numpy as np, pandas as pd, sys, pickle
from scipy.optimize import least_squares
sys.path.insert(0,'work')
from model3 import *
tr=pd.read_csv('Dataset/train.csv'); sub=tr.sample(120000,random_state=11).reset_index(drop=True)
Y=build_Y(sub); ob=~np.isnan(Y)
p=Params()
p.cw=np.load('work/wind_coef.npy') if False else None
# init wind spline from raw f04/f13
v0=Y[:,COL[4]]; w0=Y[:,COL[13]]; r0=Y[:,COL[10]]
m=np.isfinite(v0)&np.isfinite(w0)&np.isfinite(r0)
B=bas(TW,np.clip(v0[m],VLO,VHI))*r0[m][:,None]; nb=B.shape[1]
D2=np.diff(np.eye(nb),2,axis=0); lam=0.01*len(B)/nb
p.cw=np.linalg.solve(B.T@B+lam*(D2.T@D2), B.T@w0[m])

for rd in range(3):
    th,_=map_latents(Y,p,n_iter=20)
    G,T,H,V,Pr_,C,S,D=[th[:,i] for i in range(NL)]
    rho,Tp,tau,eta,psol,pwin,pg,tl,freq=derived(th,p)
    m=ob[:,COL[13]]
    B=bas(TW,np.clip(V[m],VLO,VHI))*rho[m][:,None]
    p.cw=np.linalg.solve(B.T@B+lam*(D2.T@D2), B.T@Y[m,COL[13]])
    m=ob[:,COL[11]]
    q=least_squares(lambda q:(q[0]+q[1]*np.sqrt(np.clip(S[m],1e-9,None)))-Y[m,COL[11]],[p.e0,p.e1],method='lm').x
    p.e0,p.e1=q
    m=ob[:,COL[12]]
    q=least_squares(lambda q:q[0]*(G[m]/1000)*tau[m]*eta[m]*(1-q[1]*C[m])*(1-q[2]*H[m]/100)-Y[m,COL[12]],
                    [p.K,p.ca,p.cb],method='lm').x; p.K,p.ca,p.cb=q
    m7,m15=ob[:,COL[7]],ob[:,COL[15]]
    def thr(q):
        tp_=T+q[0]*G-q[1]*V
        return np.concatenate([(tp_[m7]-Y[m7,COL[7]])/SIG[COL[7]],
                               ((1-q[2]*(tp_[m15]-p.Tref))-Y[m15,COL[15]])/SIG[COL[15]]])
    q=least_squares(thr,[p.a1,p.a2,p.beta],method='lm').x; p.a1,p.a2,p.beta=q
    m=ob[:,COL[22]]
    q=least_squares(lambda q:np.minimum(q[2],q[0]+q[1]*psol[m])-Y[m,COL[22]],[p.l0,p.l1,p.lcap],method='lm').x
    p.l0,p.l1,p.lcap=q
    A=np.column_stack([np.ones(len(T)),T]); c=np.linalg.lstsq(A,D,rcond=None)[0]
    p.d0,p.d1=c; p.dsd=max((D-A@c).std(),0.02)
    m=ob[:,COL[21]]
    A=np.column_stack([np.ones(m.sum()),(pg-D)[m]]); c=np.linalg.lstsq(A,Y[m,COL[21]],rcond=None)[0]
    p.f0,p.fq=c
    pred=forward(th,p); R=pred-Y
    print(f'round {rd}: resid_std ' + ' '.join(f'{OBS_IDS[j]:02d}:{np.nanstd(R[:,j]):.4g}' for j in range(NO)), flush=True)

print('\na1=%.6f a2=%.6f beta=%.6f Tref=%.2f'%(p.a1,p.a2,p.beta,p.Tref))
print('K=%.5f ca=%.5f cb=%.5f  e0=%.5f e1=%.5f'%(p.K,p.ca,p.cb,p.e0,p.e1))
print('l0=%.5f l1=%.5f lcap=%.5f'%(p.l0,p.l1,p.lcap))
print('d0=%.5f d1=%.5f dsd=%.5f  f0=%.5f fq=%.5f'%(p.d0,p.d1,p.dsd,p.f0,p.fq))
gv=np.arange(0,18.5,1.0); print('wind s(v):', ' '.join(f'{a:.0f}:{b:.2f}' for a,b in zip(gv,p.wind(gv))))
pickle.dump(p, open('work/params_final.pkl','wb'))
