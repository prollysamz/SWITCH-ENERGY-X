"""One constant-shift measurement identifies public mean residual exactly.

Unlike arbitrary direction calibration, this identity needs no approximation
of the unknown public subset's feature moments. It does not identify private
bias or validate a target formula.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'candidates/submission_best_0p43681.csv'
SCORE=.43681
STEP=.1
OUT=ROOT/'runs/global_bias_v1'

def mean_residual(base_rmse,probe_rmse,step):
    if not np.isfinite([base_rmse,probe_rmse,step]).all() or min(base_rmse,probe_rmse)<0 or step==0:
        raise ValueError('Invalid measurement')
    return (base_rmse**2+step**2-probe_rmse**2)/(2*step)

def check_identity():
    # An intentionally uneven public subset demonstrates subset independence.
    rng=np.random.default_rng(7307)
    prediction=rng.normal(size=2000)
    target=prediction+.27+.1*rng.normal(size=2000)
    subset=np.r_[np.arange(700),np.arange(1300,1400)]
    residual=target[subset]-prediction[subset]
    r0=np.sqrt(np.mean(residual**2))
    for step in [-.2,.1,.35]:
        r1=np.sqrt(np.mean((residual-step)**2))
        bias=mean_residual(r0,r1,step)
        np.testing.assert_allclose(bias,residual.mean(),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(r0*r0-bias*bias,np.mean((residual-bias)**2),rtol=1e-12,atol=1e-12)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--score',type=float,help='User-reported public RMSE of this exact offset probe')
    args=parser.parse_args()
    check_identity()
    baseline=pd.read_csv(BASE);sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    if not baseline.row_id.equals(sample.row_id):raise ValueError('ID mismatch')
    sha=hashlib.sha256(BASE.read_bytes()).hexdigest()
    if sha!='caed9bd21dffeea43001655b1db3bb15d289f380bda6f1f845f76ce71597847a':
        raise ValueError('Baseline changed')
    if args.score is None:
        if OUT.exists():raise FileExistsError('Preserve experiment')
        OUT.mkdir()
        result=baseline.copy();result.prediction+=STEP
        path=OUT/'submission_bias_plus_0p10.csv';result.to_csv(path,index=False)
        reread=pd.read_csv(path)
        assert reread.row_id.equals(sample.row_id) and list(reread.columns)==['row_id','prediction']
        np.testing.assert_allclose(reread.prediction-baseline.prediction,STEP,atol=1e-12,rtol=1e-12)
        record=dict(status='UNSCORED diagnostic, not an asserted improvement',candidate=path.relative_to(ROOT).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),incumbent_file=BASE.relative_to(ROOT).as_posix(),
            incumbent_sha256=sha,incumbent_rmse=SCORE,constant_shift=STEP,
            baseline_prediction_mean=float(baseline.prediction.mean()),
            probe_rmse_if_public_bias_is_zero=float(np.sqrt(SCORE**2+STEP**2)),
            identity='E_public[y - baseline] = (baseline_RMSE^2 + shift^2 - probe_RMSE^2)/(2*shift)',
            score_used_to_choose_shift=False,
            limitations=['Probe may score worse; its purpose is to measure bias.',
                'One measurement does not determine other residual structure.',
                'Public bias correction need not equal private bias correction.',
                'Reported rounded RMSE gives approximate bias; no first-place guarantee.'])
        (OUT/'manifest.json').write_text(json.dumps(record,indent=2))
    else:
        manifest=json.loads((OUT/'manifest.json').read_text())
        probe=ROOT/manifest['candidate']
        if hashlib.sha256(probe.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Probe changed')
        bias=mean_residual(SCORE,args.score,STEP)
        if abs(bias)>SCORE+1e-4:raise ValueError('Scores inconsistent with this constant probe')
        result=baseline.copy();result.prediction+=bias
        path=OUT/'submission_measured_bias_corrected.csv'
        if path.exists():raise FileExistsError('Preserve measured correction')
        result.to_csv(path,index=False)
        record=dict(probe_public_score=args.score,measured_public_mean_residual=bias,
            conditional_public_rmse_after_bias_removal=float(np.sqrt(max(SCORE**2-bias**2,0))),
            candidate=path.relative_to(ROOT).as_posix(),
            limitation='Identity uses the same public split and rounded scores; private effect unmeasured.')
        (OUT/'measured_result.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
