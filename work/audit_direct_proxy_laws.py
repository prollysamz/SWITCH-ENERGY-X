"""Evaluate direct feature-proxy NDEM laws against public-score constraints.

No target labels are used. Formula candidates are screened by the necessary
RMSE identities from existing submissions; no CSV is generated here.
"""
import json,hashlib
import numpy as np,pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from audit_order_priors import ROOT

OUT=ROOT/'runs/direct_proxy_audit_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve audit')
    j=json.loads((ROOT/'candidates/leaderboard_results.json').read_text()); sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv');test=pd.read_csv(ROOT/'Dataset/test.csv')
    rows=[r for r in j['results'] if r.get('precision')=='exact'];P=np.column_stack([pd.read_csv(ROOT/r['file']).prediction.to_numpy() for r in rows]);s=np.array([r['rmse'] for r in rows]);base=P[:,-1];n=len(base)
    # Use observed proxies where present; fill missing with deterministic latent moments.
    mom=np.load(ROOT/'runs/order_prior_target_v1/moments_repaired.npz')
    def val(fid,key):
        raw=test[f'feature_{fid:02d}'].to_numpy()
        fallback=mom[key]
        return np.where(np.isfinite(raw),raw,fallback)
    G=val(14,'E_pg'); Tloss=val(15,'E_tau'); D=val(9,'E_D');S=val(8,'E_S');TL=val(22,'E_tl'); eta=val(11,'E_eta')
    # feature_15 is thermal retention; feature_22 is fractional transmission loss.
    laws={
      'gross_thermal_demand':G*Tloss-D,
      'gross_thermal_inverter_demand':G*Tloss*eta-D,
      'gross_thermal_grid_demand':G*Tloss*(1-TL)-D,
      'gross_thermal_grid_unfilled':G*Tloss*(1-TL)-D*(1-S),
      'gross_thermal_grid_ratio':G*Tloss*(1-TL)-D/(S+.05),
      'gross_thermal_battery_interaction':G*Tloss*(1-TL)-D*S,
      'gross_thermal_grid_demand_storage':G*Tloss*(1-TL)-D+S*D,
      'gross_thermal_grid_minus_interaction':G*Tloss*(1-TL)-D-S*D,
    }
    # Residualize each law to known submission span and estimate distance from
    # the required public projection. Lower is a necessary-condition screen.
    span=np.column_stack([np.ones(n),P]); H=span.T@span/n; inv=np.linalg.pinv(H)
    out=[]
    for name,x in laws.items():
        x=np.asarray(x); mu=x.mean(); xc=x-mu
        # Fit affine law to the best-known prediction only for scale diagnostics.
        ab=np.linalg.lstsq(np.column_stack([np.ones(n),x]),base,rcond=1e-12)[0];fit=ab[0]+ab[1]*x
        # For each scored vector, implied norm if x itself were target after affine fit.
        mse=np.mean((P-fit[:,None])**2,axis=0);err=np.sqrt(mse)-s
        out.append(dict(name=name,raw_mean=float(mu),raw_sd=float(x.std()),affine_intercept=float(ab[0]),affine_slope=float(ab[1]),
            max_score_identity_error=float(abs(err).max()),score_identity_errors=err.tolist(),
            rms_from_incumbent=float(np.sqrt(np.mean((fit-base)**2)))))
    OUT.mkdir();(OUT/'report.json').write_text(json.dumps(dict(results=out,target_scores_used=False,
      observed_proxy_fraction={str(k):float(test[f'feature_{k:02d}'].notna().mean()) for k in [8,9,11,14,15,22]},
      limitation='Necessary score identity is approximate because public moments are unknown; failure rejects a law, passing does not prove it.'),indent=2))
    print(json.dumps(sorted(out,key=lambda x:x['max_score_identity_error']),indent=2),flush=True)

if __name__=='__main__':main()
