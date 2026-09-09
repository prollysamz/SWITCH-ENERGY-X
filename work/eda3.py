import pandas as pd, numpy as np
tr = pd.read_csv('Dataset/train.csv')
f = [f'feature_{i:02d}' for i in range(1,26)]
C = tr[f].corr()
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
print('=== correlation matrix (x100) ===')
print((C*100).round(0).astype(int).to_string())
