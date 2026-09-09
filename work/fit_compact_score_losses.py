"""Fit compact energy laws to measured loss values with nonnegative residual variance.

This replaces unconstrained score-moment inversion, which can imply impossible
negative variance. Coefficients are not interpreted causally. Recent demand and
storage feedback remains outside fitting. No submission is generated here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr
from build_hypotheses import expected_drawdown

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/compact_loss_fit_v1'


def main():
    if OUT.exists():raise FileExistsError('Preserve experiment')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    rows=ledger['results'];sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    frames=[pd.read_csv(ROOT/r['file']) for r in rows]
    if any(not f.row_id.equals(sample.row_id) for f in frames):raise ValueError('IDs differ')
    P=np.column_stack([f.prediction for f in frames]);S=np.array([r['rmse'] for r in rows])
    excluded=['observed_thermal_v1','deterministic_v1','learned_energy_v1','direct_demand_v1','storage_contribution_v1','temperature_slope_v1']
    fit=np.array([not any(x in r['file'] for x in excluded) and r.get('precision')!='approximate'
        and 'approximate' not in r.get('note','') and r['file']!='candidates/submission_thermal_tail.csv' for r in rows])
    z=np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    spec=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    T=(1-z['E_tau'])/spec['beta']+spec['tref'];d=T-25;var=z['V_tau']/spec['beta']**2
    sd=np.maximum(np.sqrt(var),1e-9);a=d/sd
    warm=(d*d+var)*ndtr(a)+d*sd*np.exp(-a*a/2)/np.sqrt(2*np.pi)
    batteries={'ratio':expected_drawdown(z),'unfilled_demand':z['E_D']*(1-z['E_S']), 'net_demand':z['E_D']-z['E_S']}
    reports=[];outputs={}
    base=pd.read_csv(ROOT/ledger['best_file']).prediction.to_numpy()
    for battery,dd in batteries.items():
        for shape,heat in [('warm',warm),('symmetric',d*d+var)]:
            X=np.column_stack([z['E_pg']-z['solar'],z['solar'],dd,heat,z['pg_panel'],T])
            means=X.mean(0);X-=means
            H=X.T@X/len(X);cross=X.T@(P-3.9)/len(X);p2=np.mean((P-3.9)**2,axis=0)
            def predictions(theta):
                c=theta[:6];variance=theta[6]
                return np.sqrt(np.maximum(p2-2*c@cross+c@H@c+variance,1e-12))
            def residual(theta):return predictions(theta)[fit]-S[fit]
            solutions=[]
            for b0 in [-.1,0,.1]:
                theta=[.58,2.5,b0,-.008,.001,0.,.1]
                sol=least_squares(residual,theta,bounds=([.1,0,-5,-.03,-.02,-.5,0], [2,10,5,0,.02,.5,2]),
                    x_scale=[.5,2,1,.01,.01,.1,.1],max_nfev=3000,ftol=1e-12,xtol=1e-12,gtol=1e-12)
                if sol.success and np.isfinite(sol.x).all():solutions.append(sol)
            if not solutions:raise ValueError('Loss fit failed')
            best=min(solutions,key=lambda s:np.sum(residual(s.x)**2));forecast=predictions(best.x)
            pred=3.9+X@best.x[:6];name=battery+'_'+shape
            rec=dict(name=name,coefficients=best.x[:6].tolist(),residual_variance=float(best.x[6]),
                fitting_rmse_discrepancy=float(np.sqrt(np.mean((forecast[fit]-S[fit])**2))),
                max_fitting_discrepancy=float(np.max(np.abs(forecast[fit]-S[fit]))),
                rms_change_from_best=float(np.sqrt(np.mean((pred-base)**2))),
                coef_names=['wind','solar','battery','heat','generation_temperature','linear_temperature'],
                checks=[dict(file=r['file'],fitted=bool(fit[i]),reported=r['rmse'],forecast=float(forecast[i])) for i,r in enumerate(rows)])
            reports.append(rec);outputs[name]=pred
            print(json.dumps({k:v for k,v in rec.items() if k!='checks'}),flush=True)
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(dict(status='Loss-fitting diagnostic, no submission.',models=reports,
        limitations=['Assumes target equals compact law plus independent residual.', 'Public moments approximated with full test rows.',
                    'Residual variance is a model parameter, not an observed noise floor.','Fixed target mean 3.9 is approximate.',
                    'Signs of battery coefficients are unconstrained because hard negative constraints forced earlier fits to zero.']),indent=2))
    np.savez_compressed(OUT/'diagnostic_predictions.npz',row_id=sample.row_id.to_numpy(),**outputs)


if __name__=='__main__':main()
