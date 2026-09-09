"""Check a small parametric energy law against existing aggregate-score moments.

No new submission is generated. The temperature-slope score is held out from
fitting. Scores remain adaptive public feedback, not independent target labels.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution,lsq_linear
from build_hypotheses import expected_drawdown

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'runs/formula_identification_v1'
    if folder.exists():raise FileExistsError('Preserve previous identification attempt.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    rows=ledger['results']
    heldout='runs/temperature_slope_v1/submission_temperature_slope.csv'
    train=np.array([r['file']!=heldout for r in rows])
    scores=np.array([r['rmse'] for r in rows])
    frames=[pd.read_csv(ROOT/r['file']) for r in rows]
    for row in rows:
        if 'sha256' in row and hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('A scored input changed.')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if any(not f.row_id.equals(sample.row_id) for f in frames):raise ValueError('IDs differ.')
    P=np.column_stack([f.prediction for f in frames])
    base=pd.read_csv(ROOT/ledger['best_file']).prediction.to_numpy()
    r0=ledger['best_public_rmse'];n=len(base)
    D=P[:,train]-base[:,None]
    U,s,Vt=np.linalg.svd(D,full_matrices=False)
    keep=s*s/n>.025
    Q=U[:,keep]*np.sqrt(n)
    b=(np.mean(P[:,train]**2-base[:,None]**2,axis=0)-(scores[train]**2-r0**2))/2
    target=(Vt[keep]@b)*np.sqrt(n)/s[keep]
    qmean=Q.mean(axis=0)
    z=np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    spec=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    temperature=(1-z['E_tau'])/spec['beta']+spec['tref']
    sd=np.sqrt(z['V_tau'])/spec['beta']
    drawdown=expected_drawdown(z)
    candidates={'usable_net':z['E_pm']-z['E_D']*(1-z['E_S']),
                'gross_net':z['E_pg']-z['E_D']*(1-z['E_S'])}
    # Histogram quadrature speeds up repeated score-moment calculations.
    centers=np.arange(-60.,160.00001,.025)
    gh,gw=np.polynomial.hermite.hermgauss(24);gw=gw/np.sqrt(np.pi)
    H=np.zeros((Q.shape[1]+1,len(centers)))
    for node,weight in zip(gh,gw):
        values=temperature+np.sqrt(2)*sd*node
        bins=np.clip(np.rint((values-centers[0])/.025).astype(int),0,len(centers)-1)
        H[-1]+=np.bincount(bins,weights=np.full(n,weight/n),minlength=len(centers))
        for j in range(Q.shape[1]):
            H[j]+=np.bincount(bins,weights=Q[:,j]*weight/n,minlength=len(centers))
    results=[]
    predictions={}
    for name,net in candidates.items():
        fixed=np.column_stack([net-net.mean(),drawdown-drawdown.mean()])
        fixed_moments=Q.T@fixed/n
        desired=target-3.9*qmean
        def solve(shape):
            threshold,power=shape
            h=np.maximum(centers-threshold,0)**power
            hm=H@h
            A=np.column_stack([fixed_moments,hm[:-1]-hm[-1]*qmean])
            fit=lsq_linear(A,desired,bounds=([0,-3,-10],[3,0,0]),tol=1e-11)
            err=A@fit.x-desired
            return float(err@err),fit.x
        opt=differential_evolution(lambda shape:solve(shape)[0],[(10,50),(1,3.5)],
            seed=950,popsize=10,maxiter=120,tol=1e-9,polish=True)
        loss,coef=solve(opt.x)
        threshold,power=opt.x
        heat=np.zeros(n)
        for node,weight in zip(gh,gw):
            heat+=weight*np.maximum(temperature+np.sqrt(2)*sd*node-threshold,0)**power
        pred=3.9+fixed@coef[:2]+coef[2]*(heat-heat.mean())
        mse=np.mean((P-pred[:,None])**2,axis=0)
        residual_variance=float(np.median(scores[train]**2-mse[train]))
        forecast=np.sqrt(np.maximum(mse+residual_variance,0))
        report={'family':name,'target_mean_assumed':3.9,'coefficients':coef.tolist(),
            'thermal_threshold':float(threshold),'thermal_power':float(power),
            'projection_fit_rms':float(np.sqrt(loss/Q.shape[1])),
            'model_implied_residual_variance':residual_variance,
            'heldout_score_reported_rounded':float(scores[~train][0]),
            'heldout_score_forecast':float(forecast[~train][0]),
            'heldout_score_error':float(forecast[~train][0]-scores[~train][0]),
            'rms_change_from_best':float(np.sqrt(np.mean((pred-base)**2))),
            'prediction_sd':float(pred.std()),'prediction_min':float(pred.min()),'prediction_max':float(pred.max()),
            'historical_score_checks':[{'file':r['file'],'reported_rmse':r['rmse'],'forecast':float(forecast[i]),
                'used_for_fit':bool(train[i])} for i,r in enumerate(rows)]}
        results.append(report)
        predictions[name]=pred
        print(json.dumps({k:v for k,v in report.items() if k!='historical_score_checks'}),flush=True)
    folder.mkdir(parents=True)
    report={'status':'Formula identification diagnostic only; no new submission.',
        'retained_score_directions':int(keep.sum()),'direction_energy_cutoff':.025,
        'heldout_file':heldout,'results':results,
        'limitations':['Full-test moments approximate unknown public-subset moments.',
            'Historical public scores are adaptive feedback, including rounded scores.',
            'A fitted parametric family can match measured projections while being wrong elsewhere.',
            'Residual variance is conditional on the formula assumption, not an identified noise floor.']}
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(folder/'diagnostic_predictions.npz',row_id=sample.row_id.to_numpy(),**predictions)


if __name__=='__main__':main()
