import pandas as pd, numpy as np
tr = pd.read_csv('Dataset/train.csv')
F = {i: tr[f'feature_{i:02d}'].values for i in range(1,26)}
v, w, rho = F[4], F[13], F[10]
m = np.isfinite(v)&np.isfinite(w)
print('f13 max=%.3f  f04 max=%.3f  f04 q999=%.3f'%(np.nanmax(w), np.nanmax(v), np.nanquantile(v,0.999)))
print('\n--- top of the curve: f13 by f04 bins (upper tail, no clipping) ---')
X,Y = v[m], w[m]
for lo,hi in [(9,9.5),(9.5,10),(10,10.5),(10.5,11),(11,11.5),(11.5,12),(12,12.5),(12.5,13),(13,14),(14,15),(15,16),(16,18),(18,25)]:
    s=(X>=lo)&(X<hi)
    if s.sum()<10: continue
    print(f'  v in [{lo},{hi}): n={s.sum():6d} mean_v={X[s].mean():6.3f} mean_P={Y[s].mean():8.4f} sd={Y[s].std():7.4f} max={Y[s].max():8.4f}')
# does dividing by rho tighten it?
m2 = m & np.isfinite(rho)
print('\n--- spread of f13 vs f13/rho within narrow v bins ---')
X2,Y2,R2_ = v[m2], w[m2], rho[m2]
for lo,hi in [(4,4.2),(6,6.2),(8,8.2),(10,10.2)]:
    s=(X2>=lo)&(X2<hi)
    if s.sum()<50: continue
    a=Y2[s]; b=Y2[s]/R2_[s]
    print(f'  v[{lo},{hi}) n={s.sum():5d}  cv(P)={a.std()/a.mean():.4f}   cv(P/rho)={b.std()/b.mean():.4f}')
# implied exponent: log P vs log v
k = m & (v>1.5) & (w>0.05)
lv, lw = np.log(v[k]), np.log(w[k])
print('\n--- local log-log slope d(logP)/d(logv) ---')
e = np.quantile(lv, np.linspace(0,1,26)); idx=np.clip(np.searchsorted(e,lv,'right')-1,0,25)
prev=None
for b in range(26):
    s=idx==b
    if s.sum()<200: continue
    mv, mw = lv[s].mean(), lw[s].mean()
    if prev is not None:
        print(f'   v~{np.exp((mv+prev[0])/2):7.3f}   slope={(mw-prev[1])/(mv-prev[0]):7.4f}')
    prev=(mv,mw)
