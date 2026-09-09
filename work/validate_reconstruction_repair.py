"""Compare reconstruction on fresh held-out test sensors, without NDEM scores."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import build_Y,derived,map_latents as old_map
from reconstruction_repair import RepairedParams,map_latents as new_map

ROOT=Path(__file__).resolve().parents[1]


def metrics(error):
    return {'n':len(error),'rmse':float(np.sqrt(np.mean(error**2))),
            'mae':float(np.mean(abs(error))),'bias':float(error.mean()),
            'max_abs':float(np.max(abs(error)))}


def main():
    df=pd.read_csv(ROOT/'Dataset/test.csv')
    original=pickle.load(open(ROOT/'work/params_final.pkl','rb'))
    repaired=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    # Exclude rows sampled in prior observed-sensor audits, plus the known
    # regression example. The regression row has a separate deterministic test.
    excluded=set(df[df.feature_14.notna()].sample(8000,random_state=908).row_id)
    excluded.update(df[df.feature_07.notna()].sample(5000,random_state=910).row_id)
    excluded.add(438963)
    fresh=df[~df.row_id.isin(excluded)]
    scenarios=[('panel_missing',7,[7]),('panel_and_loss_missing',7,[7,15]),
               ('panel_loss_ambient_missing',7,[7,15,2]),('gross_missing',14,[14]),
               ('gross_frequency_missing',14,[14,21]),('wind_generation_missing',13,[13])]
    rows=[]
    for label,target,hide in scenarios:
        eligible=fresh[fresh[f'feature_{target:02d}'].notna()]
        ordinary=eligible.sample(min(6000,len(eligible)),random_state=921)
        # Deliberately retain rare high-wind cases, and report them separately.
        high=eligible[(eligible.feature_04>14)|(eligible.feature_18>14**2)]
        subset=pd.concat([ordinary,high]).drop_duplicates('row_id').reset_index(drop=True)
        truth=subset[f'feature_{target:02d}'].to_numpy()
        masks={'all':np.ones(len(subset),bool),
               'wind_above14':((subset.feature_04>14)|(subset.feature_18>14**2)).to_numpy(),
               'wind_above18':((subset.feature_04>18)|(subset.feature_18>18**2)).to_numpy(),
               'ordinary_wind':~((subset.feature_04>14)|(subset.feature_18>14**2)).to_numpy(),
               'high_heat':(subset.feature_02>35).to_numpy(),
               'low_storage':(subset.feature_08<.25).to_numpy()}
        masked=subset.copy()
        for fid in hide:
            masked[f'feature_{fid:02d}']=np.nan
        Y=build_Y(masked)
        for name,p,mapper in [('original',original,old_map),('repair',repaired,new_map)]:
            if name=='repair':
                th,_,info=mapper(Y,p,return_info=True)
                info={k:int(v.sum()) for k,v in info.items() if v.dtype==bool}
            else:
                th,_=mapper(Y,p)
                info={}
            derived_values=derived(th,p)
            estimate=derived_values[{7:1,13:5,14:6}[target]]
            for regime,mask in masks.items():
                if mask.sum():
                    rec={'scenario':label,'target_sensor':target,'hidden':hide,
                         'method':name,'regime':regime,**metrics((estimate-truth)[mask]),
                         'fit_diagnostics':info}
                    rows.append(rec)
                    if regime in ['all','wind_above18']:
                        print(json.dumps(rec),flush=True)
    report={'note':'Sensor errors only, not target RMSE. No target coefficients tuned.',
            'excluded_prior_audit_row_count':len(excluded),'seed':921,'results':rows}
    output=ROOT/'work/reconstruction_repair_validation.json'
    if output.exists():
        raise FileExistsError('Validation results already exist; preserve the first evaluation.')
    output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
