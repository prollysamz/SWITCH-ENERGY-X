"""Search likely NumPy RNG seeds/strides for exact distractor fingerprints."""
import json
import numpy as np,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/interleaved_rng_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve run')
    x=pd.read_csv(ROOT/'Dataset/train.csv',usecols=['feature_20','feature_25']).iloc[:100]
    obs={k:x[k].to_numpy() for k in x}
    # Compare finite first 12 values against strided normal streams for likely seeds.
    hits=[];best=[]
    for engine in ['default_rng','RandomState']:
      for seed in range(5000):
        rng=np.random.default_rng(seed) if engine=='default_rng' else np.random.RandomState(seed)
        stream=rng.standard_normal(250000)
        for stride in range(1,65):
          for offset in range(stride):
            for name,y in obs.items():
              finite=np.isfinite(y[:12]);pred=stream[offset:offset+stride*12:stride]
              if len(pred)!=12:continue
              err=float(np.max(abs(pred[finite]-y[:12][finite])))
              best.append((err,engine,seed,stride,offset,name))
              if err<1e-8:hits.append(dict(engine=engine,seed=seed,stride=stride,offset=offset,sensor=name,error=err))
    best.sort(key=lambda z:z[0]);rec=dict(hits=hits,best=[dict(error=a,engine=b,seed=c,stride=d,offset=e,sensor=f) for a,b,c,d,e,f in best[:20]],
      searched_seeds=5000,max_stream_draws=250000,strides='1..64',target_data_accessed=False)
    OUT.mkdir();(OUT/'report.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
