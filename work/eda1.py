import pandas as pd, numpy as np
tr = pd.read_csv('Dataset/train.csv')
te = pd.read_csv('Dataset/test.csv')
print('train', tr.shape, 'test', te.shape)
f = [c for c in tr.columns if c.startswith('feature')]
d = pd.read_csv('Dataset/data_dictionary.csv')
desc = dict(zip(d.feature, d.description))
cat = dict(zip(d.feature, d.category))
rows=[]
for c in f:
    s = tr[c]
    rows.append(dict(feat=c, cat=cat[c], miss=s.isna().mean(), mn=s.min(), q01=s.quantile(.01), q25=s.quantile(.25),
                     med=s.median(), q75=s.quantile(.75), q99=s.quantile(.99), mx=s.max(), mean=s.mean(), std=s.std(),
                     skew=s.skew(), nuniq=s.nunique(), desc=desc[c]))
st = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50)
print(st.to_string(float_format=lambda x: f'{x:,.4g}'))
print()
print('overall missing frac train:', tr[f].isna().mean().mean(), 'test:', te[f].isna().mean().mean())
print('min observed per row train:', tr[f].notna().sum(1).min(), 'max', tr[f].notna().sum(1).max())
print('obs count distribution:'); print(tr[f].notna().sum(1).value_counts().sort_index())
