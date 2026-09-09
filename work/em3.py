import numpy as np, pandas as pd, sys, pickle
from scipy.optimize import least_squares
sys.path.insert(0,'work')
from model2 import *

tr = pd.read_csv('Dataset/train.csv')
sub = tr.sample(80000, random_state=7).reset_index(drop=True)
Y = build_Y(sub); ob = ~np.isnan(Y); Yf = np.nan_to_num(Y,nan=0.0)
col = {fid:j for j,fid in enumerate(OBS_IDS)}
sd = np.nanstd(Y,axis=0)
sig = 0.05*sd; sig[col[4]] = 0.24; sig[col[13]] = 0.05; sig[col[14]] = 0.05; sig[col[21]] = 0.07
p = Params()
# init splines from raw data
v0 = np.nan_to_num(Y[:,col[4]], nan=5.3); r0 = np.nan_to_num(Y[:,col[10]], nan=1.176)
p.cw = np.linalg.lstsq(basis(TW, np.clip(v0,-1,22))*r0[:,None],
                       np.nan_to_num(Y[:,col[13]],nan=4.4), rcond=None)[0]
s0 = np.nan_to_num(Y[:,col[8]], nan=0.554)
p.ce = np.linalg.lstsq(basis(TE, np.clip(s0,0,1.1)), np.nan_to_num(Y[:,col[11]],nan=0.947), rcond=None)[0]

def refit(th, p, sig):
    G,T,H,V,Pr_,C,S,D = [th[:,i] for i in range(NL)]
    rho,Tp,tau,eta,psol,pwin,pg,tl,freq = derived(th,p)
    # wind spline
    m = ob[:,col[13]]
    Bw = basis(TW, np.clip(V[m],-1,22))*rho[m][:,None]
    p.cw = np.linalg.lstsq(Bw, Y[m,col[13]], rcond=None)[0]
    # eta spline
    m = ob[:,col[11]]
    p.ce = np.linalg.lstsq(basis(TE, np.clip(S[m],0,1.1)), Y[m,col[11]], rcond=None)[0]
    # solar params K, ca, cb
    m = ob[:,col[12]]
    def f12r(q):
        return q[0]*(G[m]/1000)*tau[m]*eta[m]*(1-q[1]*C[m])*(1-q[2]*H[m]/100) - Y[m,col[12]]
    q = least_squares(f12r,[p.K,p.ca,p.cb],method='lm').x; p.K,p.ca,p.cb = q
    # thermal a1,a2,beta from f07 & f15
    m7, m15 = ob[:,col[7]], ob[:,col[15]]
    def thr(q):
        tp_ = T + q[0]*G - q[1]*V
        r7 = (tp_[m7]-Y[m7,col[7]])/sig[col[7]]
        r15 = ((1-q[2]*(tp_[m15]-p.Tref))-Y[m15,col[15]])/sig[col[15]]
        return np.concatenate([r7,r15])
    q = least_squares(thr,[p.a1,p.a2,p.beta],method='lm').x; p.a1,p.a2,p.beta = q
    # transmission loss
    m = ob[:,col[22]]
    def tlr(q): return np.minimum(q[2], q[0]+q[1]*psol[m]) - Y[m,col[22]]
    q = least_squares(tlr,[p.l0,p.l1,p.lcap],method='lm').x; p.l0,p.l1,p.lcap = q
    # demand | temp
    A = np.column_stack([np.ones(len(T)), T]); c = np.linalg.lstsq(A, D, rcond=None)[0]
    p.d0,p.d1 = c; p.dsd = max((D - A@c).std(), 0.02)
    # grid frequency
    m = ob[:,col[21]]
    A = np.column_stack([np.ones(m.sum()), (pg-D)[m]]); c = np.linalg.lstsq(A, Y[m,col[21]], rcond=None)[0]
    p.f0,p.fq = c
    return p

for em in range(7):
    W = 1.0/sig
    th, cov, r = map_latents(Y, W, p, n_iter=16, need_cov=True)
    # leverage-corrected sigma update
    NR = r.shape[1]; n=len(Y); J = np.empty((n,NR,NL))
    for k in range(NL):
        h=1e-5*PRIOR_SD[k]; tp=th.copy(); tp[:,k]+=h
        J[:,:,k]=(resid_fn(tp,Yf,ob,W,p)-r)/h
    hd = np.einsum('njk,nkl,njl->nj', J, cov, J)
    new = sig.copy()
    for j in range(NO):
        m = ob[:,j]
        new[j] = sig[j]*np.sqrt(max(np.mean(r[m,j]**2)/max(np.mean(1-hd[m,j]),0.03), 1e-10))
    sig = np.clip(0.35*sig+0.65*new, 1e-7, None)
    p = refit(th, p, sig)
    print(f'EM{em}: '+' '.join(f'{OBS_IDS[j]:02d}:{sig[j]:.4g}' for j in range(NO)), flush=True)

print('\nparams: a1=%.6f a2=%.6f beta=%.6f K=%.5f ca=%.5f cb=%.5f'%(p.a1,p.a2,p.beta,p.K,p.ca,p.cb))
print('        l0=%.5f l1=%.5f lcap=%.5f  d0=%.5f d1=%.5f dsd=%.5f  f0=%.5f fq=%.5f'%(p.l0,p.l1,p.lcap,p.d0,p.d1,p.dsd,p.f0,p.fq))
vv=np.linspace(0,20,21); print('\nwind curve s(v) [P at rho=1]:'); print(np.round(p.wind(vv),3))
ss=np.linspace(0.05,1.0,10); print('eta(S):', np.round(p.eta(ss),5))
pickle.dump({'p':p,'sig':sig}, open('work/fit.pkl','wb'))
