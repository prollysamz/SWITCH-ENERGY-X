"""Semi-parametric structural model: physics where it's known, splines where it isn't."""
import numpy as np
from scipy.interpolate import BSpline

LAT = ['G','T','H','V','P','C','S','D']
NL = 8
OBS_IDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,21,22]
NO = len(OBS_IDS)
PRIOR_MU = np.array([718.5, 26.99, 66.89, 5.328, 1013.0, 0.3547, 0.5538, 0.6669])
PRIOR_SD = np.array([340.0,  7.05,  17.0,  2.80,  12.15, 0.182, 0.177, 0.168])

def mk_knots(lo, hi, n, k=3):
    t = np.linspace(lo, hi, n)
    return np.r_[[lo]*k, t, [hi]*k]

TW = mk_knots(-1.0, 22.0, 20)          # wind spline knots
TE = mk_knots(0.0, 1.10, 7)            # inverter-efficiency spline knots
NW = len(TW)-3-1
NE = len(TE)-3-1

def basis(t, x, k=3):
    n = len(t)-k-1
    B = np.zeros((len(x), n))
    for j in range(n):
        c = np.zeros(n); c[j] = 1.0
        B[:, j] = BSpline(t, c, k, extrapolate=True)(x)
    return B

class Params:
    def __init__(self):
        self.a1, self.a2 = 0.021952, 0.344874
        self.beta, self.Tref = 0.004181, 25.0
        self.K, self.ca, self.cb = 0.9569, 0.4099, 0.2979
        self.l0, self.l1, self.lcap = 0.02997, 0.112093, 0.079656
        self.d0, self.d1, self.dsd = 0.217357, 0.016654, 0.1194
        self.fq, self.f0 = 0.9, 50.0
        self.cw = None          # wind spline coefs
        self.ce = None          # eta spline coefs
    def wind(self, V):
        return BSpline(TW, self.cw, 3, extrapolate=True)(np.clip(V,-1.0,22.0))
    def eta(self, S):
        return BSpline(TE, self.ce, 3, extrapolate=True)(np.clip(S,0.0,1.10))

def derived(th, p):
    G,T,H,V,Pr_,C,S,D = [th[...,i] for i in range(NL)]
    rho  = Pr_*100.0/(287.05*(T+273.15))
    Tp   = T + p.a1*G - p.a2*V
    tau  = 1.0 - p.beta*(Tp - p.Tref)
    eta  = p.eta(S)
    psol = p.K*(G/1000.0)*tau*eta*(1.0-p.ca*C)*(1.0-p.cb*H/100.0)
    pwin = rho*p.wind(V)
    pg   = psol + pwin
    tl   = np.minimum(p.lcap, p.l0 + p.l1*psol)
    freq = p.f0 + p.fq*(pg - D)
    return rho,Tp,tau,eta,psol,pwin,pg,tl,freq

def forward(th, p):
    G,T,H,V,Pr_,C,S,D = [th[...,i] for i in range(NL)]
    rho,Tp,tau,eta,psol,pwin,pg,tl,freq = derived(th, p)
    return np.stack([G,T,H,V,Pr_,C,Tp,S,D,rho,eta,psol,pwin,pg,tau,G*H/100.0,V*V,S*D,freq,tl], axis=-1)

def resid_fn(th, Yf, ob, W, p):
    r = (forward(th, p) - Yf)*W*ob
    rp = (th - PRIOR_MU)/PRIOR_SD
    rp[:,7] = 0.0
    rd = ((th[:,7] - (p.d0 + p.d1*th[:,1]))/p.dsd)[:,None]
    return np.concatenate([r, rp, rd], axis=1)

def map_latents(Y, W, p, n_iter=20, need_cov=False, init=None):
    n = Y.shape[0]; ob = ~np.isnan(Y); Yf = np.nan_to_num(Y, nan=0.0)
    th = np.tile(PRIOR_MU,(n,1)).astype(np.float64) if init is None else init.copy()
    if init is None:
        for j,fid in enumerate(OBS_IDS):
            if fid in (1,2,3,4,5,6,8,9):
                k = {1:0,2:1,3:2,4:3,5:4,6:5,8:6,9:7}[fid]
                th[ob[:,j],k] = Y[ob[:,j],j]
    lam = np.full(n, 1e-3)
    r = resid_fn(th, Yf, ob, W, p); cost = (r*r).sum(1)
    idx = np.arange(NL)
    for it in range(n_iter):
        NR = r.shape[1]; J = np.empty((n,NR,NL))
        for k in range(NL):
            h = 1e-5*PRIOR_SD[k]; tp = th.copy(); tp[:,k]+=h
            J[:,:,k] = (resid_fn(tp,Yf,ob,W,p)-r)/h
        JT = np.swapaxes(J,1,2); A = JT@J; g = (JT@r[:,:,None])[:,:,0]
        ok = np.zeros(n,bool)
        for _ in range(6):
            Ad = A.copy(); Ad[:,idx,idx] *= (1.0+lam[:,None]); Ad[:,idx,idx] += 1e-10
            step = np.linalg.solve(Ad, -g[:,:,None])[:,:,0]
            cand = th+step; rc = resid_fn(cand,Yf,ob,W,p); cc=(rc*rc).sum(1)
            imp = (cc<cost)&~ok
            th[imp]=cand[imp]; r[imp]=rc[imp]; cost[imp]=cc[imp]
            lam[imp]*=0.3; ok|=imp; lam[~ok]*=8.0
            if ok.all(): break
    if not need_cov: return th, None, r
    NR = r.shape[1]; J = np.empty((n,NR,NL))
    for k in range(NL):
        h = 1e-5*PRIOR_SD[k]; tp=th.copy(); tp[:,k]+=h
        J[:,:,k] = (resid_fn(tp,Yf,ob,W,p)-r)/h
    A = np.swapaxes(J,1,2)@J; A[:,idx,idx]+=1e-9
    return th, np.linalg.inv(A), r

def build_Y(df):
    return np.column_stack([df[f'feature_{i:02d}'].values for i in OBS_IDS]).astype(np.float64)
