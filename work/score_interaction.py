"""Record the next measured score and calibrate this one model effect.

Usage: python work/score_interaction.py EXACT_RMSE --experiment runs/interaction_v1
Never call with a forecast. Same-public-subset/full-test-moment assumptions apply.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tune_scored_pair import optimal_weight,line_mse

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('score',type=float)
    parser.add_argument('--experiment',default='runs/interaction_v1')
    parser.add_argument('--approximate',action='store_true',help='Record rounded feedback without fitting coefficients.')
    args=parser.parse_args()
    if not np.isfinite(args.score) or args.score<0:
        raise ValueError('A measured finite nonnegative RMSE is required.')
    folder=(ROOT/args.experiment).resolve()
    if not folder.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Experiment must be inside this workspace runs directory.')
    manifest=json.loads((folder/'manifest.json').read_text())
    source=ROOT/manifest['candidate']
    base=ROOT/manifest['incumbent_file']
    if sha(source)!=manifest['sha256'] or sha(base)!=manifest['incumbent_sha256']:
        raise ValueError('A scored input file changed.')
    p0,p1=pd.read_csv(base),pd.read_csv(source)
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    for frame in (p0,p1):
        if list(frame.columns)!=['row_id','prediction'] or not frame.row_id.equals(sample.row_id):
            raise ValueError('Submission schema or IDs differ from the sample.')
        if not frame.row_id.is_unique or not np.isfinite(frame.prediction).all():
            raise ValueError('Invalid submission values.')
    ledger_path=ROOT/'candidates/leaderboard_results.json'
    ledger=json.loads(ledger_path.read_text())
    relative=source.relative_to(ROOT).as_posix()
    previous=[r for r in ledger['results'] if Path(r['file']).as_posix()==relative]
    if previous and previous[0]['rmse']!=args.score:
        raise ValueError('A different score is already recorded for this exact file.')
    if not previous:
        ledger['results'].append({'file':relative,'rmse':args.score,'sha256':sha(source),
                                  'source':'User-reported rounded public RMSE' if args.approximate else 'User-reported exact public RMSE',
                                  'precision':'approximate' if args.approximate else 'exact'})
    if not args.approximate and args.score<ledger['best_public_rmse']:
        # Keep all reported decimal places; do not round a six-decimal score
        # into a five-decimal snapshot name or collide with a different result.
        from decimal import Decimal
        score_text=format(Decimal(str(args.score)), 'f')
        whole, _, fraction=score_text.partition('.')
        score_text=whole+'.'+fraction.ljust(5,'0')
        best=ROOT/'candidates'/('submission_best_'+score_text.replace('.','p')+'.csv')
        payload=source.read_bytes()
        if best.exists() and best.read_bytes()!=payload:
            raise FileExistsError('A different best snapshot already occupies this path.')
        best.write_bytes(payload)
        ledger['best_file']=best.relative_to(ROOT).as_posix()
        ledger['best_public_rmse']=args.score
    ledger_path.write_text(json.dumps(ledger,indent=2))
    if args.approximate:
        report={'measured_score_rounded':args.score,
                'decision':'Recorded rounded feedback; no coefficient calibration or new submission.'}
        (folder/'measured_result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return
    if manifest.get('allow_score_calibration') is False:
        report={'measured_score':args.score,
                'decision':'Recorded measured score; learned-model weights remain frozen.',
                'best_public_rmse':ledger['best_public_rmse']}
        (folder/'measured_result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return
    d=(p1.prediction-p0.prediction).to_numpy()
    q=float(np.mean(d*d))
    public_n=int(.3*len(d))
    q_se=float(np.sqrt(.7*np.var(d*d,ddof=1)/public_n))
    r0=manifest['incumbent_rmse']
    weight=optimal_weight(r0,args.score,q)
    current_t=1. if args.score<r0 else 0.
    nominal=np.sqrt(max(line_mse(weight,r0,args.score,q),0))
    improvement=min(r0,args.score)-nominal
    report={'measured_score':args.score,'original_best_score':r0,'direction_q_full_test':q,
            'nominal_weight':weight,'nominal_rmse_forecast':float(nominal),
            'forecast_improvement':float(improvement),
            'random_public_subset_q_standard_error':q_se,
            'q_stress_forecasts':[],
            'note':'Forecast, not private validation; public subset moment is unknown.'}
    for stress_q in [max(q-3*q_se,1e-12),q+3*q_se]:
        report['q_stress_forecasts'].append({'q':stress_q,
            'rmse_at_nominal_weight':float(np.sqrt(max(line_mse(weight,r0,args.score,stress_q),0)))})
    if improvement>.015:
        # Retain a small amount of the measured incumbent against moment error.
        used=current_t+.9*(weight-current_t)
        output=folder/'submission_calibrated.csv'
        if output.exists():
            raise FileExistsError('Do not overwrite a previously generated calibrated candidate.')
        result=p0.copy()
        result['prediction']=p0.prediction+used*d
        if not np.isfinite(result.prediction).all():
            raise ValueError('Nonfinite output.')
        result.to_csv(output,index=False)
        report.update({'candidate':output.relative_to(ROOT).as_posix(),
                       'weight_used':used,'sha256':sha(output),
                       'candidate_rmse_forecast':float(np.sqrt(max(line_mse(used,r0,args.score,q),0)))})
    else:
        report['decision']='Do not spend a submission on a negligible same-direction adjustment.'
    (folder/'measured_result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
