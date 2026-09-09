"""Tune one direction using two reported RMSEs and unlabelled predictions.

For p(t)=p0+t*(p1-p0), on the SAME scoring rows:
MSE(t)=MSE0+t*(MSE1-MSE0-Q)+t*t*Q, Q=mean((p1-p0)**2).
The public row IDs are unknown. Full-test Q is an approximation, not validation.
No target moments or target labels are used in this step.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def line_mse(weight, rmse0, rmse1, q):
    return rmse0**2+weight*(rmse1**2-rmse0**2-q)+weight**2*q


def optimal_weight(rmse0, rmse1, q):
    if not np.isfinite([rmse0,rmse1,q]).all() or min(rmse0,rmse1)<0 or q<=0:
        raise ValueError('Finite nonnegative scores and a positive direction moment are required.')
    if q < (rmse0-rmse1)**2-1e-8 or q > (rmse0+rmse1)**2+1e-8:
        raise ValueError('Scores and direction moment are inconsistent on a common set of rows.')
    return .5+(rmse0**2-rmse1**2)/(2*q)


def main():
    folder=ROOT/'candidates'
    ledger=json.loads((folder/'leaderboard_results.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    scores={row['file']:row['rmse'] for row in ledger['results']}
    frames=[]
    source_info=[]
    for name in ['battery','combined']:
        relative=f'candidates/submission_{name}.csv'
        path=ROOT/relative
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==manifest['candidates'][name]['sha256'], 'Scored source file changed'
        frame=pd.read_csv(path)
        assert list(frame.columns)==['row_id','prediction']
        assert frame.row_id.is_unique and np.isfinite(frame.prediction).all()
        frames.append(frame)
        source_info.append({'file':relative,'rmse':scores[relative],'sha256':digest})
    p0,p1=frames
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    assert len(p0)==100000 and np.array_equal(p0.row_id,p1.row_id)
    assert np.array_equal(p0.row_id,sample.row_id)
    r0,r1=[x['rmse'] for x in source_info]
    direction=(p1.prediction-p0.prediction).to_numpy()
    squared=direction**2
    q=float(squared.mean())
    # Finite-population SE assumes a simple random 30% public subset. The 3-SE
    # stress value is NOT a hard bound or a private leaderboard confidence interval.
    public_n=int(.3*len(squared))
    se=float(np.sqrt((1-public_n/len(squared))*squared.var(ddof=1)/public_n))
    stress_q=q+3*se
    weight=optimal_weight(r0,r1,stress_q)
    pred=p0.prediction.to_numpy()+weight*direction
    result=p0.copy()
    result['prediction']=pred
    path=folder/'submission_tuned_heat.csv'
    result.to_csv(path,index=False)
    saved=pd.read_csv(path)
    assert saved.shape==(100000,2) and saved.row_id.is_unique
    assert np.array_equal(saved.row_id,sample.row_id)
    assert np.isfinite(saved.prediction).all()
    assert np.allclose(saved.prediction,pred,rtol=1e-12,atol=1e-12)
    report={'status':'UNSCORED; aggregate-score extrapolation, not target validation',
        'sources':source_info,'formula':'battery + weight * (combined - battery)',
        'weight':weight,'nominal_optimal_weight':optimal_weight(r0,r1,q),
        'direction_second_moment_full_test':q,'public_n_assumed':public_n,
        'random_public_subset_moment_standard_error':se,'stress_second_moment':stress_q,
        'forecast_rmse_using_full_test_moment':float(np.sqrt(line_mse(weight,r0,r1,q))),
        'forecast_rmse_using_stress_moment':float(np.sqrt(line_mse(weight,r0,r1,stress_q))),
        'limitations':['Public scoring row IDs are unknown; full-test moments may differ.',
                      'The 3-SE stress calculation assumes random public row selection.',
                      'Forecasts do not include private distribution shift or score provenance errors.',
                      'No clipping is applied, because clipping changes the scored direction.'],
        'output':{'file':str(path.relative_to(ROOT)),
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'n':len(pred),'mean':float(pred.mean()),'sd':float(pred.std()),
                  'min':float(pred.min()),'max':float(pred.max()),
                  'negative_fraction':float(np.mean(pred<0))}}
    (folder/'tuned_heat_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
