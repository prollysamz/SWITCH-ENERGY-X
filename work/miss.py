import pandas as pd, numpy as np
tr = pd.read_csv('Dataset/train.csv')
cols=[f'feature_{i:02d}' for i in range(1,26)]
F = {i: tr[f'feature_{i:02d}'].values for i in range(1,26)}
M = {i: np.isnan(F[i]) for i in range(1,26)}
print('missing rates:', {i: round(M[i].mean(),4) for i in range(1,26)})
# For each feature j missing, see how the mean of each observed feature k differs
print('\n=== E[f_k | f_j missing] - E[f_k | f_j observed], in units of sd(f_k) ===')
hdr = 'j\k  ' + ' '.join(f'{k:5d}' for k in range(1,26))
print(hdr)
for j in range(1,26):
    row=[]
    for k in range(1,26):
        if k==j: row.append('    .'); continue
        a = F[k][M[j] & ~M[k]]; b = F[k][~M[j] & ~M[k]]
        if len(a)<200 or len(b)<200: row.append('    ?'); continue
        d = (a.mean()-b.mean())/F[k][~M[k]].std()
        row.append(f'{d:5.2f}')
    print(f'{j:3d}  ' + ' '.join(row))
