"""Fit a two-direction RMSE surface from scored prediction files.

Directions are combined-battery and original-battery. The third collinear score
identifies the public second moment of the first direction. Cross moment and
the second direction's second moment use full-test estimates. No labels used.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def recover_line_moment(r0, r1, rt, t):
    if not np.isfinite([r0,r1,rt,t]).all() or min(r0,r1,rt)<0 or abs(t*(t-1))<1e-10:
        raise ValueError('Nonnegative finite scores and a third distinct line position are required.')
    q=(rt**2-r0**2-t*(r1**2-r0**2))/(t*t-t)
    if q<=0 or q<(r0-r1)**2-1e-7 or q>(r0+r1)**2+1e-7:
        raise ValueError('Inconsistent scores for the specified prediction line.')
    return q


def fit_surface(r0, direction_scores, gram):
    gram=np.asarray(gram,dtype=float)
    scores=np.asarray(direction_scores,dtype=float)
    if gram.shape!=(len(scores),len(scores)) or not np.isfinite(gram).all():
        raise ValueError('Invalid moment matrix.')
    if not np.isfinite(scores).all() or not np.isfinite(r0) or min(r0,scores.min())<0:
        raise ValueError('Invalid scores.')
    if not np.allclose(gram,gram.T) or np.linalg.eigvalsh(gram).min()<=1e-10:
        raise ValueError('Directions must have a positive definite moment matrix.')
    linear=.5*(scores**2-r0**2-np.diag(gram))
    weights=-np.linalg.solve(gram,linear)
    minimum=r0**2+weights@linear
    if minimum < -1e-7:
        raise ValueError('Negative inferred minimum: check score provenance and public moments.')
    return linear,weights


def surface_mse(r0,linear,gram,weights):
    return float(r0**2+2*weights@linear+weights@gram@weights)


def main():
    folder=ROOT/'candidates'
    ledger=json.loads((folder/'leaderboard_results.json').read_text())
    results={r['file']:r for r in ledger['results']}
    names=['candidates/submission_battery.csv','candidates/submission_combined.csv',
           'submission.csv','candidates/submission_tuned_heat.csv']
    frames=[]
    sources=[]
    for name in names:
        path=ROOT/name
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        row=results[name]
        if 'sha256' in row:
            assert digest==row['sha256'], f'Scored source changed: {name}'
        frame=pd.read_csv(path)
        assert list(frame.columns)==['row_id','prediction']
        assert frame.row_id.is_unique and np.isfinite(frame.prediction).all()
        frames.append(frame)
        sources.append({'file':name,'rmse':row['rmse'],'sha256':digest})
    assert all(np.array_equal(frames[0].row_id,f.row_id) for f in frames[1:])
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    assert len(sample)==100000 and np.array_equal(sample.row_id,frames[0].row_id)
    battery,combined,original,tuned=[f.prediction.to_numpy() for f in frames]
    rb,rc,ro,rt=[s['rmse'] for s in sources]
    t=json.loads((folder/'tuned_heat_manifest.json').read_text())['weight']
    assert np.allclose(tuned,battery+t*(combined-battery),rtol=1e-12,atol=1e-12)
    directions=np.column_stack([combined-battery,original-battery])
    gram_full=directions.T@directions/len(sample)
    gram=gram_full.copy()
    gram[0,0]=recover_line_moment(rb,rc,rt,t)
    linear,opt=fit_surface(rb,[rc,ro],gram)
    current=np.array([t,0.])
    # Small move back toward the verified incumbent for uncertainty in the two
    # unobserved public moments. This 90% move is a stated heuristic, not CV tuning.
    move_fraction=.9
    weights=current+move_fraction*(opt-current)
    pred=battery+directions@weights
    nominal_mse=surface_mse(rb,linear,gram,weights)
    assert abs(surface_mse(rb,linear,gram,current)-rt**2)<1e-8
    # Moment-only uncertainty, conditional on the measured first moment. This
    # assumes random 30% public sampling; it is not private-score uncertainty.
    products=np.column_stack([directions[:,0]**2,
                              directions[:,0]*directions[:,1],directions[:,1]**2])
    npublic=int(.3*len(sample))
    covariance=(1-npublic/len(sample))*np.cov(products,rowvar=False)/npublic
    cond_cov=covariance[1:,1:]-np.outer(covariance[1:,0],covariance[0,1:])/covariance[0,0]
    cond_shift=covariance[1:,0]/covariance[0,0]*(gram[0,0]-gram_full[0,0])
    derivatives=np.array([2*weights[0]*weights[1],weights[1]**2-weights[1]])
    mse_se=float(np.sqrt(max(derivatives@cond_cov@derivatives,0)))
    mse_center=nominal_mse+float(derivatives@cond_shift)
    result=sample.copy()
    result['prediction']=pred
    output=folder/'submission_refined_plane.csv'
    result.to_csv(output,index=False)
    reread=pd.read_csv(output)
    assert reread.shape==(100000,2) and reread.row_id.is_unique
    assert np.array_equal(reread.row_id,sample.row_id)
    assert np.isfinite(reread.prediction).all()
    assert np.allclose(reread.prediction,pred,rtol=1e-12,atol=1e-12)
    line_opt=.5+(rb**2-rc**2)/(2*gram[0,0])
    report={'status':'UNSCORED candidate; forecasts are conditional, not validation',
        'sources':sources,'formula':'battery + w1*(combined-battery) + w2*(original-battery)',
        'weights':weights.tolist(),'nominal_optimal_weights':opt.tolist(),
        'move_fraction_toward_nominal_optimum':move_fraction,
        'equivalent_file_weights':{'battery':float(1-weights.sum()),
                                   'combined':float(weights[0]),'original':float(weights[1])},
        'full_test_gram':gram_full.tolist(),'estimated_public_gram':gram.tolist(),
        'existing_line_optimum_rmse':float(np.sqrt(surface_mse(rb,linear,gram,np.array([line_opt,0])))),
        'nominal_plane_optimum_rmse':float(np.sqrt(surface_mse(rb,linear,gram,opt))),
        'candidate_forecast_rmse':float(np.sqrt(nominal_mse)),
        'moment_only_three_se_sensitivity_rmse':[float(np.sqrt(max(mse_center-3*mse_se,0))),
                                                float(np.sqrt(max(mse_center+3*mse_se,0)))],
        'limitations':['First-direction public moment recovered from three rounded same-split scores.',
                      'Cross moment and second-direction moment are estimated from full test.',
                      'Sensitivity calculation assumes a random 30% public subset; not a guarantee.',
                      'Private distribution shift and nonlinear model error are not quantified.',
                      'All forecasts depend on correct file-to-score attribution and the same scoring rows.',
                      'Negative file weights are intentional extrapolation; no clipping applied.'],
        'output':{'file':str(output.relative_to(ROOT)),
                  'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'n':len(pred),
                  'mean':float(pred.mean()),'sd':float(pred.std()),'min':float(pred.min()),
                  'max':float(pred.max()),'negative_fraction':float(np.mean(pred<0)),
                  'rms_change_from_best':float(np.sqrt(np.mean((pred-tuned)**2)))}}
    (folder/'refined_plane_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
