import numpy as np, pandas as pd, sys, pickle, time
sys.path.insert(0,'work')
from model3 import *
p=pickle.load(open('work/params_final.pkl','rb'))
NS=40; CH=25000
rng=np.random.default_rng(2024)

def process(df, tag):
    Y=build_Y(df); n=len(Y)
    names=['pg','tau','eta','tl','S','D']
    E={k:np.zeros(n) for k in names}          # posterior means
    V={k:np.zeros(n) for k in names}          # posterior variances
    EC={c:np.zeros(n) for c in [16.,18.,20.,22.]}   # E[min(c, pg*tau*eta*los)]
    Epm=np.zeros(n)                            # E[pg*tau*eta*los]
    t0=time.time()
    for a in range(0,n,CH):
        b=min(a+CH,n); Yc=Y[a:b]
        th,cov=map_latents(Yc,p,n_iter=22,need_cov=True)
        L=np.linalg.cholesky(cov+np.eye(NL)*1e-12)
        acc={k:0.0 for k in names}; acc2={k:0.0 for k in names}
        accc={c:0.0 for c in EC}; accpm=0.0
        for s in range(NS):
            z=rng.standard_normal((b-a,NL))
            ts=th+np.einsum('nij,nj->ni',L,z)
            ts[:,0]=np.clip(ts[:,0],0,1300); ts[:,3]=np.clip(ts[:,3],0,25)
            ts[:,5]=np.clip(ts[:,5],0,1.05);  ts[:,6]=np.clip(ts[:,6],0.005,1.05)
            rho,Tp,tau,eta,psol,pwin,pg,tl,freq=derived(ts,p)
            S=ts[:,6]; D=ts[:,7]
            cur=dict(pg=pg,tau=tau,eta=eta,tl=tl,S=S,D=D)
            for k in names: acc[k]=acc[k]+cur[k]/NS; acc2[k]=acc2[k]+cur[k]**2/NS
            pm=pg*tau*eta*(1-tl); accpm=accpm+pm/NS
            for c in EC: accc[c]=accc[c]+np.minimum(c,pm)/NS
        for k in names: E[k][a:b]=acc[k]; V[k][a:b]=np.maximum(acc2[k]-acc[k]**2,0)
        for c in EC: EC[c][a:b]=accc[c]
        Epm[a:b]=accpm
        if a % 100000==0: print(f'  {tag} {a}/{n}  {time.time()-t0:.0f}s', flush=True)
    out={('E_'+k):E[k] for k in names}
    out.update({('V_'+k):V[k] for k in names})
    out['E_pm']=Epm
    for c in EC: out[f'E_cap{int(c)}']=EC[c]
    np.savez_compressed(f'work/blocks_{tag}.npz', **out)
    sd=np.sqrt(V['pg'])
    print(f'{tag}: n={n}  posterior sd of pg: mean={sd.mean():.4f} med={np.median(sd):.4f} '
          f'p90={np.quantile(sd,.9):.4f} p99={np.quantile(sd,.99):.4f} max={sd.max():.3f}', flush=True)
    return out

te=pd.read_csv('Dataset/test.csv');  process(te,'test')
tr=pd.read_csv('Dataset/train.csv'); process(tr,'train')
