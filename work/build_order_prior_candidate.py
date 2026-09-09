"""Apply locked local priors across physics branches of the frozen incumbent.

Legacy Monte Carlo deviations are preserved through common-moment differences.
No target coefficient is selected using new leaderboard feedback.
"""
import json
import hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_order_priors import ROOT
from frozen_current_target import matrix
from order_prior_target_impact import features

OUT=ROOT/'runs/order_prior_candidate_v1'

def design(old,rep,joint,detold,detrep,observed,spec,learned,missing):
    legacy=matrix(old,rep,joint,spec['legacy_frozen'],spec['beta'],spec['tref'])
    raw_observed=np.where(np.isfinite(observed),legacy[:,-1]-np.maximum(observed-25,0)**2,0.)
    panel=(1-rep['E_tau'])/spec['beta']+spec['tref']
    return np.column_stack([legacy,raw_observed,panel,learned,missing,
        rep['E_pm']-rep['E_D'],features(detold,detrep,spec)])

def main():
    if OUT.exists():raise FileExistsError('Preserve run')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    def read(path):
        frame=pd.read_csv(ROOT/path)
        assert frame.row_id.equals(sample.row_id)
        return frame.prediction.to_numpy()
    bestpath=ROOT/'candidates/submission_best_0p44348.csv'
    target=read(bestpath)
    spec=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    old=dict(np.load(ROOT/'work/blocks_test.npz'))
    rep=dict(np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz'))
    joint=np.load(ROOT/'runs/interaction_v1/joint_features.npz')['pg_panel']
    det=[dict(np.load(ROOT/f'runs/deterministic_v1/moments_{name}.npz')) for name in ['original','repaired']]
    new=[dict(np.load(ROOT/f'runs/order_prior_target_v1/moments_{name}.npz')) for name in ['original','repaired']]
    observed=pd.read_csv(ROOT/'Dataset/test.csv').feature_07.to_numpy()
    learned=read('runs/learned_energy_v1/submission_learned_energy.csv')-read('runs/deterministic_v1/submission_deterministic.csv')
    missing=read('runs/missingness_correction_v1/submission_missingness_correction.csv')-read('runs/successful_components_v3/submission_successful_components_v3.csv')
    X=design(old,rep,joint,*det,observed,spec,learned,missing)
    fit=sample.row_id.to_numpy()%5!=0
    scale=X[fit].std(0);scale[scale<1e-12]=1
    coef=np.linalg.lstsq(X[fit]/scale,target[fit],rcond=1e-12)[0]/scale
    error=X@coef-target
    if abs(error).max()>1e-8:raise ValueError(f'Incomplete exact algebra: {abs(error).max()}')
    updated=[];clipped={}
    for name,mc,before,after in zip(['original','repaired'],[old,rep],det,new):
        z={k:v.copy() for k,v in mc.items()}
        for k in z:
            if k!='row_id' and k in before and k in after:
                z[k]+=after[k]-before[k]
                if k.startswith('V_'):
                    clipped[name+'_'+k]=int((z[k]<0).sum())
                    z[k]=np.maximum(z[k],0)
        updated.append(z)
    newjoint=joint+new[1]['pg_panel']-det[1]['pg_panel']
    Xnew=design(*updated,newjoint,*new,observed,spec,learned,missing)
    delta=(Xnew-X)@coef
    # An alternative algebra fit on the disjoint partition must give the same update.
    alt=np.linalg.lstsq(X[~fit]/scale,target[~fit],rcond=1e-12)[0]/scale
    stability=float(abs((Xnew-X)@(alt-coef)).max())
    if stability>1e-7:raise ValueError('Nonidentifiable update')
    if not np.isfinite(delta).all():raise ValueError('Nonfinite update')
    OUT.mkdir()
    output=OUT/'submission_order_priors.csv'
    candidate=sample.copy();candidate['prediction']=target+delta
    candidate.to_csv(output,index=False)
    reread=pd.read_csv(output)
    assert reread.row_id.equals(sample.row_id) and list(reread.columns)==['row_id','prediction']
    np.testing.assert_allclose(reread.prediction,target+delta,rtol=1e-12,atol=1e-12)
    np.savez_compressed(OUT/'frozen_algebra.npz',coefficients=coef,row_id=sample.row_id,delta=delta)
    report=dict(status='UNSCORED local-prior reconstruction candidate; unchanged target coefficients',
        candidate=output.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        incumbent_file=bestpath.relative_to(ROOT).as_posix(),incumbent_score=.44348,
        incumbent_sha256=hashlib.sha256(bestpath.read_bytes()).hexdigest(),
        max_reproduction_error=float(abs(error).max()),unused_partition_max_reproduction_error=float(abs(error[~fit]).max()),
        disjoint_algebra_update_max_difference=stability,
        rms_prediction_change=float(np.sqrt(np.mean(delta**2))),max_prediction_change=float(abs(delta).max()),
        delta_quantiles=np.quantile(delta,[0,.001,.01,.5,.99,.999,1]).tolist(),
        variance_clamps=clipped,target_coefficients_refitted_to_scores=False,target_rmse_forecast=None,
        sensor_audits=['runs/order_prior_audit_v1/report.json','runs/order_prior_confirmation_v1/report.json',
                      'runs/order_prior_cross_sensor_v1/report.json'],
        limitations=['Sensor holdouts are not hidden-target validation.',
            'Gaussian prior replacement is approximate; existing Monte Carlo deviations are retained.',
            'Learned residual and missingness correction remain frozen.',
            'Observed-sensor masks may differ from natural missingness.',
            'Rank and public/private score improvements are unverified.'])
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=6):main()
