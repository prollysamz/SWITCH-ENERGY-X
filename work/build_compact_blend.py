"""Export a guarded 25% compact-law experiment, retaining 75% of scored best.

The compact law is selected using adaptive public feedback. Its coefficients
are medians across moment-stress fits; no hidden-label accuracy is asserted.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_hypotheses import expected_drawdown

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/compact_blend_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():raise FileExistsError('Preserve candidate')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    baseline_path=ROOT/ledger['best_file'];baseline=pd.read_csv(baseline_path)
    expected='ec2c74e5732c8c3d6b8f992ccccbdb76b2b242adb6841c18c5b3614604c145f3'
    if sha(baseline_path)!=expected:raise ValueError('This experiment requires the preserved 0.53504 baseline')
    stress=json.loads((ROOT/'runs/compact_loss_fit_v1/sensitivity.json').read_text())
    if stress['selected_model']!='ratio_symmetric' or len(stress['coefficients'])!=40:
        raise ValueError('Unexpected stress result')
    coefficients=np.median(np.asarray(stress['coefficients'])[:, :6],axis=0)
    sp=json.loads((ROOT/'runs/deterministic_v1/frozen_target.json').read_text())
    z=np.load(ROOT/'runs/deterministic_v1/moments_repaired.npz')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if not baseline.row_id.equals(sample.row_id) or not np.array_equal(z['row_id'],sample.row_id):
        raise ValueError('IDs differ')
    T=(1-z['E_tau'])/sp['beta']+sp['tref'];variance=z['V_tau']/sp['beta']**2
    X=np.column_stack([z['E_pg']-z['solar'],z['solar'],expected_drawdown(z),
        (T-25)**2+variance,z['pg_panel'],T])
    means=X.mean(0)
    law=3.9+(X-means)@coefficients
    delta=.25*(law-baseline.prediction.to_numpy())
    candidate=sample.copy();candidate['prediction']=baseline.prediction.to_numpy()+delta
    if not np.isfinite(candidate.prediction).all() or not candidate.row_id.is_unique:
        raise ValueError('Invalid candidate')
    if len(candidate)!=100000 or np.sqrt(np.mean(delta**2))>.20:
        raise ValueError('Candidate exceeds the planned 0.20 RMS change budget')
    OUT.mkdir()
    output=OUT/'submission_compact_blend.csv';candidate.to_csv(output,index=False)
    reread=pd.read_csv(output)
    if list(reread.columns)!=['row_id','prediction'] or not reread.row_id.equals(sample.row_id):
        raise ValueError('Round-trip schema mismatch')
    np.testing.assert_allclose(reread.prediction,candidate.prediction,atol=1e-12,rtol=1e-12)
    # Check reproduction with fixed centering and arbitrary row order.
    reproduced=3.9+(X[::-1]-means)@coefficients
    np.testing.assert_allclose(reproduced[::-1],law,atol=1e-12,rtol=1e-12)
    np.save(OUT/'direction.npy',delta)
    np.savez_compressed(OUT/'law_predictions.npz',row_id=sample.row_id.to_numpy(),prediction=law)
    q=np.sort(delta**2)
    sources=[]
    for row in ledger['results']:
        path=ROOT/row['file'];digest=sha(path)
        if 'sha256' in row and digest!=row['sha256']:raise ValueError('Scored source changed')
        sources.append(dict(file=row['file'],sha256=digest,score=row['rmse'],precision=row.get('precision','see ledger')))
    record=dict(status='Unscored 25% compact-model blend; not validated against hidden labels.',
        candidate=output.relative_to(ROOT).as_posix(),sha256=sha(output),
        incumbent_file=baseline_path.relative_to(ROOT).as_posix(),incumbent_sha256=sha(baseline_path),
        incumbent_rmse=.53504,allow_score_calibration=True,blend_weight=.25,
        coefficient_names=['wind','solar','demand_storage_ratio','symmetric_quadratic_temperature','generation_temperature','linear_temperature'],
        coefficients=coefficients.tolist(),feature_centers=means.tolist(),target_mean_assumed=3.9,
        selection='Coordinatewise median coefficients from 40 random-public-moment stress fits; 75% incumbent retained.',
        scores_excluded_from_coefficient_fit=['approximate scores','observed-temperature probe','deterministic correction',
            'learned correction','direct-demand probe','direct-storage probe','temperature-slope probe'],
        selection_uses_public_diagnostics=True,source_stress_summary=stress['summary'],
        rms_change=float(np.sqrt(np.mean(delta**2))),max_absolute_change=float(np.max(np.abs(delta))),
        top_one_percent_change_energy_share=float(q[-1000:].sum()/q.sum()),
        q_random_public_subset_se=float(np.sqrt(.7*np.var(delta**2,ddof=1)/30000)),
        target_rmse_forecast=None,sources=sources,
        code_sha256={f:sha(ROOT/'work'/f) for f in ['fit_compact_score_losses.py','stress_compact_law.py','build_compact_blend.py']},
        limitations=['Target law is inferred from adaptive aggregate feedback, not fitted to known labels.',
            'Residual-variance boundary and subset sensitivity prevent trusting the full law or a score forecast.',
            'Some older approximate scores are poorly explained; their source precision/provenance is incomplete.',
            'The ratio coefficient is positive in these fits, contrary to a simple drawdown interpretation; coefficients are not established physical effects.',
            'Taking median coefficients and a 25% blend mitigates sensitivity but does not prove generalization.'])
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ['sources','code_sha256']},indent=2))


if __name__=='__main__':main()
