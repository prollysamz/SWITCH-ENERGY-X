import numpy as np, pandas as pd, sys, json
sys.path.insert(0,'work')
from model import *

rng = np.random.default_rng(0)
tr = pd.read_csv('Dataset/train.csv')
sub = tr.sample(60000, random_state=1).reset_index(drop=True)
Y = build_Y(sub)
sd = np.nanstd(Y, axis=0)
sig = 0.05*sd                      # initial guess
sig[OBS_IDS.index(4)] = 0.26
p = dict(P0)

for em in range(6):
    W = 1.0/sig
    th, cov = map_latents(Y, W, p, PRIOR_MU, PRIOR_SD, n_iter=18)
    pred = forward(th, p)
    R = (pred - Y)                        # raw residuals
    obs = ~np.isnan(Y)
    # leverage correction via Gauss-Newton hat diagonal
    r = np.nan_to_num(R,nan=0.0)*W*obs
    news = sig.copy()
    for j in range(NO):
        rj = r[obs[:,j], j]
        news[j] = sig[j]*np.sqrt(max(np.mean(rj**2),1e-8))
    # damp updates
    sig = 0.5*sig + 0.5*np.clip(news, 1e-6, None)
    tot = np.nanmean(np.abs(np.nan_to_num(R,nan=0)*obs).sum(1))
    print(f'EM {em}: sigma = ' + ', '.join(f'f{OBS_IDS[j]:02d}={sig[j]:.4g}' for j in range(NO)), flush=True)

np.save('work/sigma.npy', sig)
json.dump({k:float(v) for k,v in p.items()}, open('work/params.json','w'), indent=1)
print('\nlatent means:', dict(zip(LAT, np.round(th.mean(0),4))))
print('latent stds :', dict(zip(LAT, np.round(th.std(0),4))))
np.save('work/th_sub.npy', th); sub.to_csv('work/sub60k.csv', index=False)
