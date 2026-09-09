"""Final structural model. Physics-parametric + penalized-spline wind curve."""
import numpy as np
from scipy.interpolate import BSpline

LAT=['G','T','H','V','P','C','S','D']; NL=8
OBS_IDS=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,21,22]; NO=len(OBS_IDS)
COL={fid:j for j,fid in enumerate(OBS_IDS)}
PRIOR_MU=np.array([718.5,26.99,66.89,5.328,1013.0,0.3547,0.5538,0.6669])
PRIOR_SD=np.array([340.0,7.05,17.0,2.80,12.15,0.182,0.177,0.168])
# sigma solved from the identity network (see notes)
SIG=np.array([19.0,0.50,1.20,0.25,2.50,0.030,0.68,0.018,0.017,0.00766,
              0.0072,0.022,0.030,0.032,0.0063,12.7,0.84,0.0158,0.058,0.0052])

def mk(lo,hi,n,k=3):
    t=np.linspace(lo,hi,n); return np.r_[[lo]*k,t,[hi]*k]
def bas(t,x,k=3):
    n=len(t)-k-1; B=np.zeros((len(x),n))
    for j in range(n):
        c=np.zeros(n); c[j]=1.0; B[:,j]=BSpline(t,c,k,extrapolate=True)(x)
    return B
TW=mk(0.0,18.0,15)
VLO,VHI=0.0,18.0

class Params:
    def __init__(s):
        s.a1,s.a2=0.021957,0.334338; s.beta,s.Tref=0.004213,25.0
        s.K,s.ca,s.cb=1.00598,0.47126,0.31306
        s.e0,s.e1=0.913869,0.045083
        s.l0,s.l1,s.lcap=0.02769,0.12020,0.07965
        s.d0,s.d1,s.dsd=0.20900,0.01699,0.115
        s.f0,s.fq=50.0,0.9
        s.cw=None
    def wind(s,V):
        return BSpline(TW,s.cw,3,extrapolate=True)(np.clip(V,VLO,VHI))
    def eta(s,S):
        return s.e0+s.e1*np.sqrt(np.clip(S,1e-9,None))

def derived(th,p):
    G,T,H,V,Pr_,C,S,D=[th[...,i] for i in range(NL)]
    rho=Pr_*100.0/(287.05*(T+273.15))
    Tp=T+p.a1*G-p.a2*V
    tau=1.0-p.beta*(Tp-p.Tref)
    eta=p.eta(S)
    psol=p.K*(G/1000.0)*tau*eta*(1.0-p.ca*C)*(1.0-p.cb*H/100.0)
    pwin=rho*p.wind(V)
    pg=psol+pwin
    tl=np.minimum(p.lcap,p.l0+p.l1*psol)
    freq=p.f0+p.fq*(pg-D)
    return rho,Tp,tau,eta,psol,pwin,pg,tl,freq

def forward(th,p):
    G,T,H,V,Pr_,C,S,D=[th[...,i] for i in range(NL)]
    rho,Tp,tau,eta,psol,pwin,pg,tl,freq=derived(th,p)
    return np.stack([G,T,H,V,Pr_,C,Tp,S,D,rho,eta,psol,pwin,pg,tau,G*H/100.0,V*V,S*D,freq,tl],axis=-1)

def resid_fn(th,Yf,ob,W,p):
    r=(forward(th,p)-Yf)*W*ob
    rp=(th-PRIOR_MU)/PRIOR_SD; rp[:,7]=0.0
    rd=((th[:,7]-(p.d0+p.d1*th[:,1]))/p.dsd)[:,None]
    return np.concatenate([r,rp,rd],axis=1)

def map_latents(Y,p,sig=None,n_iter=22,need_cov=False):
    sig=SIG if sig is None else sig
    W=1.0/sig
    n=Y.shape[0]; ob=~np.isnan(Y); Yf=np.nan_to_num(Y,nan=0.0)
    th=np.tile(PRIOR_MU,(n,1)).astype(np.float64)
    for fid,k in [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(8,6),(9,7)]:
        j=COL[fid]; th[ob[:,j],k]=Y[ob[:,j],j]
    lam=np.full(n,1e-3); idx=np.arange(NL)
    r=resid_fn(th,Yf,ob,W,p); cost=(r*r).sum(1)
    for it in range(n_iter):
        NR=r.shape[1]; J=np.empty((n,NR,NL))
        for k in range(NL):
            h=1e-5*PRIOR_SD[k]; tp=th.copy(); tp[:,k]+=h
            J[:,:,k]=(resid_fn(tp,Yf,ob,W,p)-r)/h
        JT=np.swapaxes(J,1,2); A=JT@J; g=(JT@r[:,:,None])[:,:,0]
        ok=np.zeros(n,bool)
        for _ in range(6):
            Ad=A.copy(); Ad[:,idx,idx]*=(1.0+lam[:,None]); Ad[:,idx,idx]+=1e-10
            step=np.linalg.solve(Ad,-g[:,:,None])[:,:,0]
            cand=th+step; rc=resid_fn(cand,Yf,ob,W,p); cc=(rc*rc).sum(1)
            imp=(cc<cost)&~ok
            th[imp]=cand[imp]; r[imp]=rc[imp]; cost[imp]=cc[imp]
            lam[imp]*=0.3; ok|=imp; lam[~ok]*=8.0
            if ok.all(): break
    if not need_cov: return th,None
    NR=r.shape[1]; J=np.empty((n,NR,NL))
    for k in range(NL):
        h=1e-5*PRIOR_SD[k]; tp=th.copy(); tp[:,k]+=h
        J[:,:,k]=(resid_fn(tp,Yf,ob,W,p)-r)/h
    A=np.swapaxes(J,1,2)@J; A[:,idx,idx]+=1e-9
    return th,np.linalg.inv(A)

def build_Y(df):
    return np.column_stack([df[f'feature_{i:02d}'].values for i in OBS_IDS]).astype(np.float64)
