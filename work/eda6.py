import pandas as pd, numpy as np
from scipy.optimize import least_squares
tr = pd.read_csv('Dataset/train.csv')
F = {i: tr[f'feature_{i:02d}'].values for i in range(1,26)}
def fit(y_i, build, p0, names, sub, label):
    y = F[y_i]; m = ~np.isnan(y)
    for i in sub: m &= ~np.isnan(F[i])
    Fm = {i: F[i][m] for i in sub}; ym = y[m]
    try:
        r = least_squares(lambda p: np.nan_to_num(build(p,Fm)-ym, nan=1e6), p0, method='lm', max_nfev=8000)
        rr = build(r.x,Fm)-ym
        print(f'{label:50s} n={m.sum():7d} rstd={np.nanstd(rr):.6g} R2={1-np.nanvar(rr)/ym.var():.6f}')
        print('     ', dict(zip(names, np.round(r.x,6))))
    except Exception as e: print(label, 'FAIL', e)

print('===== f12 solar generation =====', flush=True)
fit(12, lambda p,F: p[0]*(F[1]/1000)*F[15]*F[11]*(1-F[6]), [1.0], ['k'], [1,15,11,6], 'k*(f01/1000)*f15*f11*(1-f06)')
fit(12, lambda p,F: p[0]*(F[1]/1000)*F[15]*(1-F[6]), [1.0], ['k'], [1,15,6], 'k*(f01/1000)*f15*(1-f06)')
fit(12, lambda p,F: p[0]*(F[1]/1000)*F[15]*F[11], [1.0], ['k'], [1,15,11], 'k*(f01/1000)*f15*f11')
fit(12, lambda p,F: p[0]*(F[1]/1000)*F[15]*F[11]*(1-p[1]*F[6]), [1.0,1.0], ['k','c'], [1,15,11,6], 'k*(f01/1000)*f15*f11*(1-c*f06)')
fit(12, lambda p,F: p[0]*(F[1]/1000)*F[15]*F[11]*(1-F[6])*(1-p[1]*F[3]/100), [1.0,0.1], ['k','h'], [1,15,11,6,3], '...*(1-h*f03/100)')
print('===== f13 wind generation =====', flush=True)
fit(13, lambda p,F: p[0]*F[10]*F[4]**3, [0.01], ['k'], [4,10], 'k*f10*f04^3')
fit(13, lambda p,F: np.minimum(p[1], p[0]*F[10]*np.clip(F[4],0,None)**3), [0.01,22], ['k','Prated'], [4,10], 'min(Prated, k*rho*v^3)')
fit(13, lambda p,F: np.minimum(p[1], p[0]*F[10]*np.clip(F[4]-p[2],0,None)**3), [0.01,22,0.1], ['k','Prated','vin'], [4,10], 'min(Prated,k*rho*(v-vin)^3)')
fit(13, lambda p,F: 0.5*p[0]*F[10]*np.clip(F[4],0,None)**3, [0.02], ['A*Cp/1000'], [4,10], '0.5*k*rho*v^3')
print('===== f09 demand =====', flush=True)
fit(9, lambda p,F: p[0]+p[1]*F[2], [0.2,0.017], ['a','b'], [2], 'a + b*f02')
fit(9, lambda p,F: p[0]+p[1]*F[2]+p[2]*F[2]**2, [0.2,0.017,0.0], ['a','b','c'], [2], 'a + b*f02 + c*f02^2')
fit(9, lambda p,F: p[0]+p[1]*np.abs(F[2]-p[2]), [0.5,0.01,20.0], ['a','b','T0'], [2], 'a + b*|f02-T0|')
print('===== f08 SoC corr =====', flush=True)
for j in [1,2,9,12,13,14,6,24,11]:
    m=~np.isnan(F[8])&~np.isnan(F[j]); print(f'  corr(f08,f{j:02d})={np.corrcoef(F[8][m],F[j][m])[0,1]:+.4f}')
print('===== f16 precip vs f06 =====', flush=True)
fit(16, lambda p,F: p[0]+p[1]*np.clip(F[6],0,None)**p[2], [0.0,1.0,2.0], ['a','b','p'], [6], 'a + b*f06^p')
print('===== f03 humidity corr =====', flush=True)
for j in [1,2,5,6,16]:
    m=~np.isnan(F[3])&~np.isnan(F[j]); print(f'  corr(f03,f{j:02d})={np.corrcoef(F[3][m],F[j][m])[0,1]:+.4f}')
