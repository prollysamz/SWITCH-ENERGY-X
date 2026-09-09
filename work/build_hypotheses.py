"""Build distinct, UNVALIDATED target hypotheses from the competition brief.

No labels are available. Moment matching does not establish predictive accuracy.
All coefficients and standardisation are recorded; scored baseline is untouched.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]


def expected_drawdown(z, eps=0.05):
    # Integrate the convex reciprocal using a bounded marginal posterior of SoC.
    # D/S covariance is unavailable in the cached blocks: this remains approximate.
    mean = np.clip(z['E_S'], 0.05, 1.0)
    sd = np.maximum(np.sqrt(z['V_S']), 1e-5)
    a, b = ndtr((0.05-mean)/sd), ndtr((1.0-mean)/sd)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    reciprocal = np.zeros_like(mean)
    for x, w in zip(nodes, weights):
        q = np.clip(a + (x+1)*0.5*(b-a), 1e-12, 1-1e-12)
        s = np.clip(mean + sd*ndtri(q), 0.05, 1.0)
        reciprocal += 0.5*w/(s+eps)
    return np.clip(z['E_D'],0,1.5)*reciprocal


def components(z):
    pg,tau,eta,tl,S,D = [z['E_'+k] for k in ['pg','tau','eta','tl','S','D']]
    return np.array([z['E_pm']-D*(1-S), z['E_pm']-(D-S),
                     pg*tau*(1-tl)-D*(1-S), z['E_cap20']-D*(1-S),
                     z['E_pm'], pg*tau*(1-tl)-S*D])


def main():
    tr=np.load(ROOT/'work/blocks_train.npz')
    te=np.load(ROOT/'work/blocks_test.npz')
    train_c,test_c=components(tr),components(te)
    mu,sd=train_c.mean(axis=1)[:,None],train_c.std(axis=1)[:,None]
    weights=np.array([.25,.20,.20,.15,.10,.10])
    gtr=weights@((train_c-mu)/sd)
    gte=weights@((test_c-mu)/sd)
    gm,gs=gtr.mean(),gtr.std()
    gtr,gte=(gtr-gm)/gs,(gte-gm)/gs
    heat_tr=np.maximum((1-tr['E_tau'])/.004213+25-40,0)
    heat_te=np.maximum((1-te['E_tau'])/.004213+25-40,0)
    risk_tr,risk_te=expected_drawdown(tr),expected_drawdown(te)
    def residualize(a,b):
        center=float(a.mean())
        slope=float(np.mean((a-center)*gtr))
        ar=a-center-slope*gtr
        scale=float(ar.std())
        return ar/scale,(b-center-slope*gte)/scale,dict(mean=center,slope=slope,sd=scale)
    bat_tr,bat_te,bat_params=residualize(-risk_tr,-risk_te)
    hot_tr,hot_te,hot_params=residualize(-heat_tr,-heat_te)
    comb_tr,comb_te,comb_params=residualize(.8*bat_tr+.6*hot_tr,.8*bat_te+.6*hot_te)
    # Conditional calculation: assumes public target mean 3.9 and SD 4.6.
    # It is an approximate diagnostic, not calibration from known public labels.
    target_mean,target_sd,current_rmse,current_sd=3.9,4.6,2.53192,4.4
    g_scale=(target_sd**2+current_sd**2-current_rmse**2)/(2*current_sd)
    residual_scale=2.0
    sub=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    ids=pd.read_csv(ROOT/'Dataset/test.csv',usecols=['row_id']).row_id
    assert np.array_equal(ids,sub.row_id) and sub.row_id.is_unique
    baseline=pd.read_csv(ROOT/'submission.csv')
    assert np.array_equal(sub.row_id,baseline.row_id)
    out=ROOT/'candidates'; out.mkdir(exist_ok=True)
    report={'status':'UNVALIDATED target hypotheses; sensor validation is separate',
        'source':'User-provided competition screenshots and local feature data',
        'assumptions':{'target_mean':target_mean,'target_sd':target_sd,'current_rmse':current_rmse,
                       'generation_sd':g_scale,'new_effect_sd':residual_scale,
                       'soc_lower_bound':.05,'soc_upper_bound':1.,'reciprocal_epsilon':.05,
                       'heat_threshold':40.,'marginal_quadrature_nodes':24},
        'normalization':{'battery':bat_params,'heat':hot_params,'combined':comb_params},
        'scale_only_rmse_if_target_moments_exact':float(np.sqrt(target_sd**2-g_scale**2)),
        'candidates':{}}
    for name,r in [('battery',bat_te),('thermal',hot_te),('combined',comb_te)]:
        pred=target_mean+g_scale*gte+residual_scale*r
        assert len(pred)==100000 and np.isfinite(pred).all()
        sub['prediction']=pred
        path=out/f'submission_{name}.csv'
        sub.to_csv(path,index=False)
        check=pd.read_csv(path)
        assert list(check.columns)==['row_id','prediction']
        assert np.array_equal(check.row_id,ids)
        assert np.allclose(check.prediction,pred,rtol=1e-12,atol=1e-12)
        stats={'file':str(path.relative_to(ROOT)), 'mean':float(pred.mean()),'sd':float(pred.std()),
               'negative_fraction':float(np.mean(pred<0)), 'min':float(pred.min()),'max':float(pred.max()),
               'rms_change_from_baseline':float(np.sqrt(np.mean((pred-baseline.prediction)**2))),
               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        report['candidates'][name]=stats
        print(name,json.dumps(stats),flush=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print('Approximate scale-only RMSE:', report['scale_only_rmse_if_target_moments_exact'])


if __name__=='__main__':
    main()
