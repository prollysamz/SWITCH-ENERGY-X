import pandas as pd, numpy as np
from scipy.optimize import least_squares
tr = pd.read_csv('Dataset/train.csv')
F = {i: tr[f'feature_{i:02d}'].values for i in range(1,26)}

def fit(y_i, build, p0, names, sub=None, label=''):
    """build(p, F) -> prediction"""
    y = F[y_i]
    # mask: y and all used features present -> determine by evaluating with all-ones
    need = sub
    m = ~np.isnan(y)
    for i in need: m &= ~np.isnan(F[i])
    Fm = {i: F[i][m] for i in need}
    ym = y[m]
    def res(p): return build(p, Fm) - ym
    r = least_squares(res, p0, method='lm', max_nfev=20000)
    rr = res(r.x)
    print(f'{label:52s} n={m.sum():7d} resid_std={rr.std():.6g} R2={1-rr.var()/ym.var():.6f}')
    print('     params:', dict(zip(names, np.round(r.x,6))))
    return r.x

print('===== f21 grid frequency =====')
fit(21, lambda p,F: p[0]+p[1]*F[14], [50,0.9], ['a','b'], [14], 'f21 = a + b*f14')
fit(21, lambda p,F: p[0]+p[1]*(F[14]-F[9]), [50,0.9], ['a','b'], [14,9], 'f21 = a + b*(f14-f09)')
fit(21, lambda p,F: p[0]+p[1]*F[14]+p[2]*F[9], [50,0.9,-0.9], ['a','b14','b9'], [14,9], 'f21 = a + b14*f14 + b9*f09')
fit(21, lambda p,F: p[0]+p[1]*F[14]+p[2]*F[9]+p[3]*F[8]+p[4]*F[22]+p[5]*F[11], [50,.9,-.9,0,0,0],
    ['a','b14','b9','b8','b22','b11'], [14,9,8,22,11], 'f21 = a + b14 f14 + b9 f09 + b8 f08 + b22 f22 + b11 f11')
fit(21, lambda p,F: p[0]+p[1]*(F[12]+F[13]-F[9]), [50,0.9], ['a','b'], [12,13,9], 'f21 = a + b*(f12+f13-f09)')

print('===== f22 transmission loss =====')
fit(22, lambda p,F: np.minimum(p[2], p[0]+p[1]*F[12]), [0.03,0.11,0.08], ['a','b','cap'], [12], 'f22 = min(cap, a+b*f12)')
fit(22, lambda p,F: np.minimum(p[2], p[0]+p[1]*F[14]), [0.03,0.002,0.08], ['a','b','cap'], [14], 'f22 = min(cap, a+b*f14)')
fit(22, lambda p,F: np.minimum(p[3], p[0]+p[1]*F[12]+p[2]*F[9]), [0.03,0.11,0.0,0.08], ['a','b12','b9','cap'], [12,9], 'f22 = min(cap, a+b12 f12 + b9 f09)')

print('===== f11 inverter efficiency =====')
fit(11, lambda p,F: p[0]+p[1]*np.log(np.clip(F[8],1e-6,None)), [0.956,0.0145], ['a','b'], [8], 'f11 = a + b*log(f08)')
fit(11, lambda p,F: p[0]+p[1]*np.sqrt(np.clip(F[8],0,None)), [0.92,0.04], ['a','b'], [8], 'f11 = a + b*sqrt(f08)')
fit(11, lambda p,F: p[0]*(1-p[1]*np.exp(-p[2]*F[8])), [0.96,0.1,3.0], ['emax','a','b'], [8], 'f11 = emax*(1-a*exp(-b*f08))')
fit(11, lambda p,F: p[0]+p[1]*F[8]/(p[2]+F[8]), [0.92,0.05,0.1], ['a','b','c'], [8], 'f11 = a + b*f08/(c+f08)')

print('===== f15 temp loss factor =====')
fit(15, lambda p,F: 1-p[0]*(F[7]-p[1]), [0.004,25], ['beta','Tref'], [7], 'f15 = 1 - beta*(f07-Tref)')

print('===== f07 panel temp =====')
fit(7, lambda p,F: F[2]+p[0]*F[1], [0.025], ['a'], [1,2], 'f07 = f02 + a*f01')
fit(7, lambda p,F: F[2]+p[0]*F[1]/(1+p[1]*F[4]), [0.03,0.1], ['a','b'], [1,2,4], 'f07 = f02 + a*f01/(1+b*f04)')
fit(7, lambda p,F: F[2]+p[0]*F[1]-p[1]*F[4], [0.025,0.2], ['a','b'], [1,2,4], 'f07 = f02 + a*f01 - b*f04')
fit(7, lambda p,F: F[2]+p[0]*F[1]*(1-p[1]*F[4]), [0.025,0.02], ['a','b'], [1,2,4], 'f07 = f02 + a*f01*(1-b*f04)')

print('===== f10 air density =====')
fit(10, lambda p,F: F[5]*100/(p[0]*(F[2]+p[1])), [287.05,273.15], ['R','K'], [2,5], 'f10 = P*100/(R*(T+K))')
