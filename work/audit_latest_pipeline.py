"""Audit artifact integrity, optimizer convergence and temperature sampling.

Uses sensor holdouts, never claims these are hidden-target validation scores.
Preserves every model, cache and scored CSV.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import reconstruction_repair as repair
from model3 import build_Y,derived

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(pred,truth):
    err=pred-truth
    return {'n':len(err),'rmse':float(np.sqrt(np.mean(err*err))),
            'bias':float(err.mean()),'mae':float(abs(err).mean()),'max_abs':float(abs(err).max())}


def main():
    folder=ROOT/'runs/pipeline_audit_v2'
    if folder.exists():raise FileExistsError('Preserve previous audit.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    inputs={}
    for row in ledger['results']:
        path=ROOT/row['file']
        digest=sha(path)
        if 'sha256' in row and digest!=row['sha256']:raise ValueError(f'Changed file: {path}')
        frame=pd.read_csv(path)
        if list(frame.columns)!=['row_id','prediction'] or not frame.row_id.equals(sample.row_id) or not frame.row_id.is_unique or not np.isfinite(frame.prediction).all():
            raise ValueError(f'Invalid schema/IDs/values: {path}')
        inputs[row['file']]=digest
    best=ROOT/ledger['best_file']
    if sha(best)!=sha(ROOT/'runs/thermal_calibrated_v1/submission_thermal_calibrated.csv'):
        raise ValueError('Preserved best does not match scored candidate.')
    print(f'All {len(inputs)} scored CSV files passed integrity checks.',flush=True)
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    params=repair.RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    excluded=set(test[test.feature_14.notna()].sample(8000,random_state=908).row_id)
    excluded.update(test[test.feature_07.notna()].sample(5000,random_state=910).row_id)
    excluded.add(438963)
    fresh=test[~test.row_id.isin(excluded)]
    # Exclude all six earlier sensor-repair validation samples and rare-wind rows.
    for target in [7,14,13]:
        eligible=fresh[fresh[f'feature_{target:02d}'].notna()]
        excluded.update(eligible.sample(min(6000,len(eligible)),random_state=921).row_id)
        excluded.update(eligible[(eligible.feature_04>14)|(eligible.feature_18>196)].row_id)
    fresh=test[~test.row_id.isin(excluded)]
    optimizer_calls=[]
    original_solver=repair.least_squares
    def tracked_solver(*args,**kwargs):
        fit=original_solver(*args,**kwargs)
        optimizer_calls.append({'success':bool(fit.success),'status':int(fit.status),
            'nfev':int(fit.nfev),'optimality':float(fit.optimality)})
        return fit
    repair.least_squares=tracked_solver
    results=[]
    selected={}
    for label,target,hide in [('panel',7,[7]),('panel_loss',7,[7,15]),
            ('panel_loss_ambient',7,[7,15,2]),('gross_frequency',14,[14,21])]:
        eligible=fresh[fresh[f'feature_{target:02d}'].notna()]
        subset=eligible.sample(5000,random_state=932).reset_index(drop=True)
        selected[label]=subset.row_id.tolist()
        truth=subset[f'feature_{target:02d}'].to_numpy()
        masked=subset.copy()
        for fid in hide:masked[f'feature_{fid:02d}']=np.nan
        estimates={}
        for iterations in [22,66]:
            start=len(optimizer_calls)
            th,_,info=repair.map_latents(build_Y(masked),params,n_iter=iterations,return_info=True)
            estimate=derived(th,params)[1 if target==7 else 6]
            estimates[iterations]=estimate
            calls=optimizer_calls[start:]
            record={'scenario':label,'iterations':iterations,**metrics(estimate,truth),
                'fallback_calls':len(calls),'nonconverged_fallback_calls':sum(not c['success'] for c in calls),
                'accepted_fallback_rows':int(info['accepted'].sum()),
                'still_inconsistent_rows':int(info['still_inconsistent'].sum())}
            results.append(record)
            print(json.dumps(record),flush=True)
        results.append({'scenario':label,'change_22_to_66_rms':float(np.sqrt(np.mean((estimates[66]-estimates[22])**2)))})
    # The panel temperature is affine in Gaussian latents. Its posterior mean
    # before clipping is exactly the MAP-derived value; 40 random draws add noise.
    subset=fresh.sample(8000,random_state=933).reset_index(drop=True)
    th,cov,info=repair.map_latents(build_Y(subset),params,need_cov=True,return_info=True)
    a=np.array([params.a1,1.,0.,-params.a2,0.,0.,0.,0.])
    expected=derived(th,params)[1]
    variance=np.einsum('i,nij,j->n',a,cov,a)
    blocks=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    indices=pd.Index(blocks['row_id']).get_indexer(subset.row_id)
    if np.any(indices<0):raise ValueError('Cache IDs missing.')
    cached=(1-blocks['E_tau'][indices])/params.beta+params.Tref
    ordinary=~info['accepted']
    sampling={'n_ordinary':int(ordinary.sum()),
        'temperature_mean_cache_vs_analytic_rms':float(np.sqrt(np.mean((cached[ordinary]-expected[ordinary])**2))),
        'gaussian_40_draw_expected_sampling_rms':float(np.sqrt(np.mean(variance[ordinary])/40)),
        'temperature_posterior_sd_median':float(np.median(np.sqrt(variance[ordinary]))),
        'note':'Temperature units only. Sampling/clipping and posterior approximation remain distinct from target accuracy.'}
    print(json.dumps(sampling),flush=True)
    report={'status':'Completed sensor and numerical audit; not NDEM target validation.',
        'scored_files_checked':inputs,'best_rmse':ledger['best_public_rmse'],
        'excluded_previous_sensor_audit_rows':len(excluded),'sensor_holdouts':results,
        'sampling':sampling,'nonconverged_calls':[c for c in optimizer_calls if not c['success']],
        'limitations':['Observed-sensor holdouts do not reproduce all naturally missing regimes.',
            'The public target feedback already used in model selection is not an independent validation set.']}
    if any(sha(ROOT/path)!=digest for path,digest in inputs.items()):raise ValueError('Scored files changed during audit.')
    folder.mkdir(parents=True)
    (folder/'audit.json').write_text(json.dumps(report,indent=2))
    (folder/'holdout_rows.json').write_text(json.dumps(selected))
    print(f'Saved {folder}/audit.json',flush=True)


if __name__=='__main__':main()
