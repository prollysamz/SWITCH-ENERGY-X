import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'work')
from model import *

tr = pd.read_csv('Dataset/train.csv')
sub = tr.sample(60000, random_state=1).reset_index(drop=True)
Y = build_Y(sub); obs = ~np.isnan(Y)
sd = np.nanstd(Y, axis=0)
sig = 0.05*sd; sig[OBS_IDS.index(4)] = 0.26
p = dict(P0)

def hat_and_resid(Y, W, p, th):
    """returns weighted residual matrix and hat diagonal per observable"""
    n = Y.shape[0]; Yf = np.nan_to_num(Y, nan=0.0); ob = ~np.isnan(Y)
    def resid(t):
        pred = forward(t, p)
        r = (pred - Yf)*W*ob
        rp = (t - PRIOR_MU)/PRIOR_SD; rp = rp.copy(); rp[:,7] = 0.0
        rd = ((t[:,7] - (p['d0']+p['d1']*t[:,1]))/p['dsd'])[:,None]
        return np.concatenate([r, rp, rd], axis=1)
    r = resid(th); NR = r.shape[1]
    J = np.empty((n, NR, NL))
    for k in range(NL):
        h = 1e-5*PRIOR_SD[k]; tp = th.copy(); tp[:,k] += h
        J[:,:,k] = (resid(tp)-r)/h
    A = np.swapaxes(J,1,2) @ J
    A[:,np.arange(NL),np.arange(NL)] += 1e-9
    Ai = np.linalg.inv(A)
    hd = np.einsum('njk,nkl,njl->nj', J, Ai, J)
    return r, hd

for em in range(8):
    W = 1.0/sig
    th, cov = map_latents(Y, W, p, PRIOR_MU, PRIOR_SD, n_iter=16)
    r, hd = hat_and_resid(Y, W, p, th)
    news = sig.copy()
    for j in range(NO):
        m = obs[:,j]
        num = np.mean(r[m,j]**2); den = max(np.mean(1.0-hd[m,j]), 0.02)
        news[j] = sig[j]*np.sqrt(max(num/den, 1e-10))
    sig = np.clip(0.4*sig + 0.6*news, 1e-6, None)
    print(f'EM{em}: ' + ' '.join(f'f{OBS_IDS[j]:02d}={sig[j]:.4g}' for j in range(NO)), flush=True)

np.save('work/sigma.npy', sig)
print('\nmean hat diag:', ' '.join(f'f{OBS_IDS[j]:02d}={np.mean(hd[obs[:,j],j]):.3f}' for j in range(NO)))
print('latent stds :', dict(zip(LAT, np.round(th.std(0),4))))
