import pandas as pd, numpy as np
tr = pd.read_csv('Dataset/train.csv')
F = {i: tr[f'feature_{i:02d}'] for i in range(1,26)}
def rep(name, lhs, rhs):
    m = lhs.notna() & rhs.notna()
    if m.sum()<100: print(name,'too few'); return
    a, b = lhs[m].values, rhs[m].values
    r = a-b
    rel = r/np.maximum(np.abs(a),1e-9)
    print(f'{name:38s} n={m.sum():7d} corr={np.corrcoef(a,b)[0,1]:.6f} resid_std={r.std():.6g} '
          f'resid_mean={r.mean():.6g} |rel|med={np.median(np.abs(rel)):.3e} lhs_std={a.std():.6g}')

print('--- engineered ---')
rep('f17 vs f01*f03/100', F[17], F[1]*F[3]/100)
rep('f17 vs f01*f03', F[17], F[1]*F[3])
rep('f18 vs f04^2', F[18], F[4]**2)
rep('f19 vs f08*f09', F[19], F[8]*F[9])
print('--- physics ---')
rep('f10 vs P/(R T)', F[10], F[5]*100/(287.05*(F[2]+273.15)))
rep('f14 vs f12+f13', F[14], F[12]+F[13])
rep('f15 vs 1-.004(f07-25)', F[15], 1-0.004*(F[7]-25))
print('--- fit linear: f07 ~ f02 + f01 ---')
import itertools
def lsq(y, Xcols, names):
    m = y.notna()
    for c in Xcols: m &= c.notna()
    Y = y[m].values; X = np.column_stack([c[m].values for c in Xcols]+[np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X@coef; r = Y-pred
    print('  y~', names, 'coef=', np.round(coef,6), 'resid_std=%.5g'%r.std(), 'R2=%.6f'%(1-r.var()/Y.var()), 'n=',m.sum())
lsq(F[7], [F[2], F[1]], ['f02','f01'])
lsq(F[7], [F[2], F[1], F[4]], ['f02','f01','f04'])
lsq(F[15], [F[7]], ['f07'])
lsq(F[22], [F[9]], ['f09'])
lsq(F[22], [F[9], F[14]], ['f09','f14'])
lsq(F[11], [F[9]], ['f09'])
lsq(F[11], [F[12]], ['f12'])
lsq(F[11], [F[14]], ['f14'])
