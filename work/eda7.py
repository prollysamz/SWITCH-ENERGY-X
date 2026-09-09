import pandas as pd, numpy as np
tr = pd.read_csv('Dataset/train.csv')
F = {i: tr[f'feature_{i:02d}'].values for i in range(1,26)}
# ratio r = f12 / ((f01/1000)*f15*f11)
den = (F[1]/1000)*F[15]*F[11]
r = F[12]/den
m = np.isfinite(r) & (F[1]>200)
print('ratio stats: mean=%.5f std=%.5f'%(np.nanmean(r[m]), np.nanstd(r[m])))
def binstat(x, y, name, nb=20):
    k = m & np.isfinite(x) & np.isfinite(y)
    X,Y = x[k], y[k]
    e = np.quantile(X, np.linspace(0,1,nb+1)); idx=np.clip(np.searchsorted(e,X,'right')-1,0,nb-1)
    print(f'-- ratio vs {name} --')
    for b in range(nb):
        s = idx==b
        if s.sum()<50: continue
        print(f'   {name}={X[s].mean():9.4f}  ratio={Y[s].mean():8.5f} sd={Y[s].std():.5f} n={s.sum()}')
binstat(F[6], r, 'f06 cloud')
binstat(F[3], r, 'f03 humid')
binstat(F[1], r, 'f01 irr')
binstat(F[16], r, 'f16 precip')
