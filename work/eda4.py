import pandas as pd, numpy as np
tr = pd.read_csv('Dataset/train.csv')
F = {i: tr[f'feature_{i:02d}'].values for i in range(1,26)}
def binplot(x, y, name, nb=25, qlo=0.001, qhi=0.999):
    m = ~np.isnan(x) & ~np.isnan(y)
    X, Y = x[m], y[m]
    lo, hi = np.quantile(X,qlo), np.quantile(X,qhi)
    k = (X>=lo)&(X<=hi); X,Y = X[k],Y[k]
    edges = np.quantile(X, np.linspace(0,1,nb+1))
    idx = np.clip(np.searchsorted(edges, X, 'right')-1, 0, nb-1)
    print(f'--- {name} (n={m.sum()}) ---')
    out=[]
    for b in range(nb):
        s = idx==b
        if s.sum()<20: continue
        out.append((X[s].mean(), Y[s].mean(), Y[s].std()))
    for a,b,c in out: print(f'   x={a:10.4f}  ymean={b:10.5f}  ystd={c:8.5f}')

# wind power curve
binplot(F[4], F[13], 'f13 (wind gen) vs f04 (wind speed)', 30)
# f21 vs f14
binplot(F[14], F[21], 'f21 (grid freq) vs f14 (gross gen)', 25)
# f22 vs f12
binplot(F[12], F[22], 'f22 (trans loss) vs f12 (solar gen)', 25)
# f11 vs f08
binplot(F[8], F[11], 'f11 (inv eff) vs f08 (SoC)', 25)
