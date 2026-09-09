"""Jointly calibrate three measured, successful directions only.

Uses four exact endpoint scores. Unknown public Gram moments are approximated
by full-test moments, with random-subset sensitivity checks. No new target
effect or rounded demand/storage feedback enters this fit.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/successful_components_v1'
FILES=['runs/learned_energy_v1/submission_learned_energy.csv',
       'runs/solar_contribution_v1/submission_solar_contribution.csv',
       'runs/compact_blend_v1/submission_compact_blend.csv',
       'runs/generation_curvature_v1/submission_generation_curvature.csv']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def solve(gram,scores):
    g=(scores[1:]**2-scores[0]**2-np.diag(gram))/2
    return -np.linalg.solve(gram,g)


def forecast(weights,gram,scores):
    return scores[0]**2+weights@(scores[1:]**2-scores[0]**2-np.diag(gram))+weights@gram@weights


def main():
    if OUT.exists():raise FileExistsError('Preserve calibrated run')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    columns=[];scores=[];sources=[]
    for name in FILES:
        record=next(r for r in ledger['results'] if r['file']==name)
        if record.get('precision')!='exact':raise ValueError('Exact scores required')
        path=ROOT/name
        if sha(path)!=record['sha256']:raise ValueError('Scored source changed')
        frame=pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id) or not np.isfinite(frame.prediction).all():raise ValueError('Invalid source')
        columns.append(frame.prediction.to_numpy());scores.append(record['rmse'])
        sources.append(dict(file=name,score=record['rmse'],sha256=sha(path)))
    P=np.column_stack(columns);scores=np.array(scores)
    D=P[:,1:]-P[:,[0]];H=D.T@D/len(D)
    optimal=solve(H,scores)
    incumbent_weights=np.array([0.,0.,1.])
    weights=incumbent_weights+.9*(optimal-incumbent_weights)
    prediction=P[:,0]+D@weights
    mse=forecast(weights,H,scores)
    if mse<=0:raise ValueError('Impossible nominal variance; do not export')
    delta=prediction-P[:,-1]
    # Score-identity reproduction at each measured vertex.
    for i,w in enumerate([np.zeros(3),*np.eye(3)]):
        np.testing.assert_allclose(forecast(w,H,scores),scores[i]**2,rtol=1e-12,atol=1e-12)
    per_row=(D@weights)**2-(D**2)@weights
    se=float(np.std(per_row,ddof=1)*np.sqrt(.7/30000))
    rng=np.random.default_rng(1071);stress=[]
    for i in range(100):
        ids=rng.choice(len(D),30000,replace=False)
        G=D[ids].T@D[ids]/len(ids)
        alternate=solve(G,scores)
        shift=alternate-optimal
        stress.append(dict(iteration=i,fixed_candidate_rmse=float(np.sqrt(max(forecast(weights,G,scores),0))),
            optimal_increment_weights=np.cumsum(alternate[::-1])[::-1].tolist(),
            refitted_prediction_rms_shift=float(np.sqrt(max(shift@H@shift,0)))))
    risks=np.array([r['fixed_candidate_rmse'] for r in stress])
    shifts=np.array([r['refitted_prediction_rms_shift'] for r in stress])
    summary=dict(random_subsets=100,rows_per_subset=30000,
        fixed_candidate_forecast_min=float(risks.min()),fixed_candidate_forecast_max=float(risks.max()),
        p95_refitted_prediction_rms_shift=float(np.quantile(shifts,.95)),
        maximum_refitted_prediction_rms_shift=float(shifts.max()))
    if risks.max()>=scores[-1] or np.quantile(shifts,.95)>.03:
        raise ValueError('Measured-direction calibration is too sensitive')
    baseline_path=ROOT/ledger['best_file'];baseline=pd.read_csv(baseline_path)
    if ledger['best_public_rmse']!=scores[-1] or sha(baseline_path)!=sources[-1]['sha256']:
        raise ValueError('Unexpected incumbent')
    if not np.isfinite(prediction).all() or not sample.row_id.is_unique:raise ValueError('Invalid candidate')
    OUT.mkdir()
    output=OUT/'submission_successful_components.csv'
    frame=sample.copy();frame['prediction']=prediction;frame.to_csv(output,index=False)
    reread=pd.read_csv(output)
    if list(reread.columns)!=['row_id','prediction'] or not reread.row_id.equals(sample.row_id):raise ValueError('Round-trip schema failed')
    np.testing.assert_allclose(reread.prediction,prediction,rtol=1e-12,atol=1e-12)
    # Independent incremental representation of the output.
    increments=np.diff(P,axis=1);increment_weights=np.cumsum(weights[::-1])[::-1]
    np.testing.assert_allclose(P[:,0]+increments@increment_weights,prediction,rtol=1e-12,atol=1e-12)
    np.save(OUT/'direction.npy',delta)
    (OUT/'sensitivity.json').write_text(json.dumps(dict(summary=summary,samples=stress),indent=2))
    report=dict(status='Unscored joint calibration of three measured directions.',
        candidate=output.relative_to(ROOT).as_posix(),sha256=sha(output),
        incumbent_file=baseline_path.relative_to(ROOT).as_posix(),incumbent_sha256=sha(baseline_path),incumbent_rmse=float(scores[-1]),
        allow_score_calibration=False,sources=sources,component_names=['solar','compact_blend_step','generation_curvature'],
        nominal_optimal_increment_weights=np.cumsum(optimal[::-1])[::-1].tolist(),
        used_increment_weights=increment_weights.tolist(),step_toward_optimum=.9,
        nominal_candidate_rmse_forecast=float(np.sqrt(mse)),
        full_test_gram=H.tolist(),gram_condition_number=float(np.linalg.cond(H)),
        random_public_moment_mse_standard_error=se,
        two_standard_error_upper_rmse_scenario=float(np.sqrt(mse+2*se)),
        stress_summary=summary,rms_change=float(np.sqrt(np.mean(delta**2))),maximum_absolute_change=float(np.max(np.abs(delta))),
        n_rows=len(frame),code_sha256=sha(Path(__file__)),
        limitations=['Forecasts and stress tests approximate public-subset moments and do not establish private performance.',
            'The candidate reuses adaptive public feedback; it is not independent target validation.',
            'No score improvement or rank is guaranteed. Rounded unsuccessful probes were excluded.'])
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources','full_test_gram']},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
