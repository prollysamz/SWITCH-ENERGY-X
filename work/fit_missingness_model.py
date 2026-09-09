"""Fit sensor-failure probabilities from independent observed environmental proxies.

Neither hidden NDEM labels nor leaderboard scores enter this estimation.
Temperature uses irradiance/ambient/wind sensors, never the panel sensor whose
failure probability is being estimated. Humidity uses its own observed proxy.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]


def probability(x,q):
    return q[0]+q[1]*expit((x-q[2])/q[3])


def fit_group(df,x,features,start,bounds):
    finite=np.isfinite(x)
    ids=df.row_id.to_numpy()[finite]
    values=x[finite]
    missing=df[[f'feature_{i:02d}' for i in features]].isna().to_numpy()[finite]
    count=missing.sum(axis=1)
    trials=len(features)
    train=ids%5!=0
    def loss(q):
        p=np.clip(probability(values[train],q),1e-9,1-1e-9)
        return -np.mean(count[train]*np.log(p)+(trials-count[train])*np.log1p(-p))
    fit=minimize(loss,start,bounds=bounds,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-13})
    if not fit.success:raise ValueError(f'Missingness fit failed: {fit.message}')
    p=np.clip(probability(values[~train],fit.x),1e-9,1-1e-9)
    constant=missing[train].mean()
    c=count[~train]
    l1=-(c*np.log(p)+(trials-c)*np.log1p(-p))/trials
    l0=-(c*np.log(constant)+(trials-c)*np.log1p(-constant))/trials
    delta=l0-l1
    return {'features':features,'coefficients':fit.x.tolist(),'formula':'p0 + amplitude * sigmoid((value-center)/scale)',
        'training_rows':int(train.sum()),'validation_rows':int((~train).sum()),
        'baseline_probability':float(constant),'heldout_baseline_logloss':float(l0.mean()),
        'heldout_conditional_logloss':float(l1.mean()),
        'logloss_improvement':float(delta.mean()),'paired_improvement_se':float(delta.std(ddof=1)/np.sqrt(len(delta))),
        'heldout_brier_baseline':float(np.mean((missing[~train]-constant)**2)),
        'heldout_brier_conditional':float(np.mean((missing[~train]-p[:,None])**2))}


def main():
    folder=ROOT/'runs/missingness_v1'
    if folder.exists():raise FileExistsError('Preserve previous experiment.')
    df=pd.read_csv(ROOT/'Dataset/train.csv')
    p=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    panel=(df.feature_02+p.a1*df.feature_01-p.a2*df.feature_04).to_numpy()
    heat=fit_group(df,panel,[7,11,15,21],[.25,.09,38.,2.],[(.05,.5),(0,.4),(20,60),(.25,15)])
    humid=fit_group(df,df.feature_03.to_numpy(),[6,16],[.25,.075,85.,2.],[(.05,.5),(0,.4),(60,100),(.25,15)])
    report={'status':'Sensor-missingness fit and held-out likelihood validation only.',
        'temperature':heat,'humidity':humid,'target_scores_used':False,
        'validation_split':'row_id modulo 5 equals zero; parameters fitted on remaining train rows',
        'limitations':['Environmental proxies contain measurement noise.',
            'Conditional independence of sensor failures is an assumption.',
            'Predicting missingness does not itself prove better imputation or target predictions.']}
    folder.mkdir(parents=True)
    (folder/'model.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
