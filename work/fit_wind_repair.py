"""Fit only the wind tail to observed TRAIN sensors; never reads target scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]


def main():
    df=pd.read_csv(ROOT/'Dataset/train.csv')
    v=np.sqrt(df.feature_18.clip(lower=0).to_numpy())
    power=(df.feature_13/df.feature_10).to_numpy()
    use=np.isfinite(v)&np.isfinite(power)&(v>=13)&(power>0)
    # Deterministic held-out training sensor rows; the full test set is not read.
    fit_mask=use&(df.row_id.to_numpy()%5!=0)
    holdout=use&~fit_mask
    def curve(v,c):
        return np.exp(c[0])*v**3*expit(-(v-c[1])/c[2])
    fit=least_squares(lambda c: curve(v[fit_mask],c)-power[fit_mask],
        [-4.,13.,2.2],loss='soft_l1',f_scale=.2,
        bounds=([-10,0,.1],[3,30,10]))
    residual=curve(v[holdout],fit.x)-power[holdout]
    report={'coefficients':fit.x.tolist(),'formula':'exp(c0)*V^3*expit(-(V-c1)/c2)',
            'fit_source':'Dataset/train.csv only; row_id % 5 != 0; observed f18,f13,f10',
            'fit_rows':int(fit_mask.sum()),'sensor_holdout_rows':int(holdout.sum()),
            'sensor_holdout_rmse':float(np.sqrt(np.mean(residual**2))),
            'no_target_scores_used':True,'join':'Old spline below 14; smoothstep join 14 to 15'}
    output=ROOT/'work/wind_repair_v2.json'
    if output.exists():
        raise FileExistsError('Versioned fit already exists; do not overwrite it.')
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
