"""Check quadratic energy formulas with the already measured generation-temperature interaction.

No submission is generated. Temperature-slope feedback is held out. Observed
temperature-noise feedback is excluded because it is not a latent energy term.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from build_hypotheses import expected_drawdown

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'runs/formula_interaction_v2'
    if folder.exists():raise FileExistsError('Preserve previous profile.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    rows=ledger['results']
    holdout='runs/temperature_slope_v1/submission_temperature_slope.csv'
    excluded='runs/observed_thermal_v1/submission_observed_temperature.csv'
    fitting=np.array([r['file'] not in [holdout,excluded] for r in rows])
    scores=np.array([r['rmse'] for r in rows])
    frames=[pd.read_csv(ROOT/r['file']) for r in rows]
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    for row,frame in zip(rows,frames):
        if 'sha256' in row and hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('A scored input changed.')
        if not frame.row_id.equals(sample.row_id) or not np.isfinite(frame.prediction).all():
            raise ValueError('Invalid scored input IDs or values.')
    P=np.column_stack([frame.prediction for frame in frames])
    base=pd.read_csv(ROOT/ledger['best_file']).prediction.to_numpy()
    n=len(base);r0=ledger['best_public_rmse']
    D=P[:,fitting]-base[:,None]
    U,s,Vt=np.linalg.svd(D,full_matrices=False)
    keep=s*s/n>.025
    Q=U[:,keep]*np.sqrt(n)
    b=(np.mean(P[:,fitting]**2-base[:,None]**2,axis=0)-(scores[fitting]**2-r0**2))/2
    target=(Vt[keep]@b)*np.sqrt(n)/s[keep]
    qmean=Q.mean(axis=0)
    z=np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    spec=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    T=(1-z['E_tau'])/spec['beta']+spec['tref']
    sd=np.sqrt(z['V_tau'])/spec['beta']
    demand=z['E_D']*(1-z['E_S'])
    drawdown=expected_drawdown(z)
    families={'gross_net':z['E_pg']-demand,
        'delivered_net':z['E_pg']*(1-z['E_tl'])-demand,
        'usable_net':z['E_pm']-demand}
    centers=np.arange(-60.,160.0001,.025)
    gh,gw=np.polynomial.hermite.hermgauss(24);gw/=np.sqrt(np.pi)
    H=np.zeros((Q.shape[1]+1,len(centers)))
    for node,weight in zip(gh,gw):
        bins=np.clip(np.rint((T+np.sqrt(2)*sd*node-centers[0])/.025).astype(int),0,len(centers)-1)
        H[-1]+=np.bincount(bins,weights=np.full(n,weight/n),minlength=len(centers))
        for j in range(Q.shape[1]):H[j]+=np.bincount(bins,weights=Q[:,j]*weight/n,minlength=len(centers))
    profiles=[]
    for name,net in families.items():
        interaction=z['pg_panel']
        fixed=np.column_stack([net-net.mean(),drawdown-drawdown.mean(),interaction-interaction.mean()])
        fixed_moments=Q.T@fixed/n
        for power in [2.]:
            best=None
            for threshold in np.arange(15.,45.01,.25):
                hm=H@np.maximum(centers-threshold,0)**power
                A=np.column_stack([fixed_moments,hm[:-1]-hm[-1]*qmean])
                fit=lsq_linear(A,target-3.9*qmean,bounds=([0,-3,-.02,-10],[3,0,.02,0]),tol=1e-10)
                loss=float(np.mean((A@fit.x-(target-3.9*qmean))**2))
                heat=np.zeros(n)
                for node,weight in zip(gh,gw):heat+=weight*np.maximum(T+np.sqrt(2)*sd*node-threshold,0)**power
                pred=3.9+fixed@fit.x[:3]+fit.x[3]*(heat-heat.mean())
                mse=np.mean((P-pred[:,None])**2,axis=0)
                residual=float(np.median(scores[fitting]**2-mse[fitting]))
                forecast=np.sqrt(np.maximum(mse+residual,0))
                heldout_index=next(i for i,r in enumerate(rows) if r['file']==holdout)
                maxerr=float(np.max(abs(forecast[fitting]-scores[fitting])))
                rec={'family':name,'power':power,'threshold':float(threshold),'coefficients':fit.x.tolist(),
                    'projection_fit_rms':float(np.sqrt(loss)),'model_implied_residual_variance':residual,
                    'maximum_fitting_score_error':maxerr,'heldout_score_forecast':float(forecast[heldout_index]),
                    'heldout_score_error':float(forecast[heldout_index]-scores[heldout_index]),
                    'rms_change_from_best':float(np.sqrt(np.mean((pred-base)**2)))}
                rec['passes_consistency_checks']=residual>=0 and maxerr<.025 and abs(rec['heldout_score_error'])<.015
                profiles.append(rec)
                # Selection criterion does not use held-out score.
                if residual>=0 and (best is None or loss<best[0]):best=(loss,rec)
            print(json.dumps({'family':name,'power':power,'best_nonnegative_residual_fit':None if best is None else best[1]}),flush=True)
    valid=[r for r in profiles if r['passes_consistency_checks']]
    report={'status':'Diagnostic profile only; no submission generated.',
        'retained_directions':int(keep.sum()),'profiles':len(profiles),
        'consistent_profile_count':len(valid),'best_consistent':sorted(valid,key=lambda r:r['projection_fit_rms'])[:10],
        'limitations':['Quadratic thermal loss and a free generation-temperature interaction are hypotheses; target mean 3.9 is assumed.',
            'Public moments and some historical scores are approximate.',
            'Held-back aggregate-score agreement is weaker than independent target validation.',
            'Passing checks would support a hypothesis, not prove its unmeasured target score.']}
    folder.mkdir(parents=True)
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    (folder/'profiles.json').write_text(json.dumps(profiles,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
