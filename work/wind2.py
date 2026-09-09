import numpy as np, pandas as pd, sys
sys.path.insert(0,'work')
from scipy.interpolate import BSpline
tr = pd.read_csv('Dataset/train.csv')
v = tr['feature_04'].values; w = tr['feature_13'].values; rho = tr['feature_10'].values
m = np.isfinite(v)&np.isfinite(w)&np.isfinite(rho)
V,W,R = v[m], w[m], rho[m]
print('n =', m.sum())
def mk(lo,hi,n,k=3):
    t=np.linspace(lo,hi,n); return np.r_[[lo]*k,t,[hi]*k]
def bas(t,x,k=3):
    n=len(t)-k-1; B=np.zeros((len(x),n))
    for j in range(n):
        c=np.zeros(n); c[j]=1.0; B[:,j]=BSpline(t,c,k,extrapolate=True)(x)
    return B
T = mk(0.0, 20.0, 16)
Xc = np.clip(V,0,20)
B = bas(T,Xc)*R[:,None]
nb = B.shape[1]
D2 = np.diff(np.eye(nb), 2, axis=0)
for lam in [0.0, 1e-3, 1e-1, 1.0, 10.0]:
    A = B.T@B + lam*(D2.T@D2)*len(V)/nb
    c = np.linalg.solve(A, B.T@W)
    r = B@c - W
    print(f'lam={lam:8.4g} resid_std={r.std():.5f}  curve@[2,5,8,11,13,15,17,19]=' +
          np.array2string(BSpline(T,c,3,extrapolate=True)(np.array([2.,5,8,11,13,15,17,19])), precision=2))
lam = 1e-1
A = B.T@B + lam*(D2.T@D2)*len(V)/nb
c = np.linalg.solve(A, B.T@W)
s = BSpline(T,c,3,extrapolate=True)
gv = np.arange(0,20.5,0.5)
print('\nfitted s(v):'); print(' '.join(f'{a:.1f}:{b:.2f}' for a,b in zip(gv, s(gv))))
# residual diagnostics: does resid depend on rho? on v?
r = B@c - W
print('\nresid std overall %.4f'%r.std())
for lo,hi in [(0,3),(3,5),(5,7),(7,9),(9,11),(11,14),(14,20)]:
    k=(V>=lo)&(V<hi); print(f'  v[{lo},{hi}) n={k.sum():6d} resid_sd={r[k].std():.4f} mean={r[k].mean():+.4f}')
np.save('work/wind_knots.npy', T); np.save('work/wind_coef.npy', c)
