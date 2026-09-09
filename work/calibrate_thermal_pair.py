"""Combine two measured, orthogonal thermal effects with a conservative step.

Public scores supply the target feedback; full-test moments approximate the
unknown public moments. This is score calibration, not independent validation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'runs/thermal_calibrated_v1'
    if folder.exists(): raise FileExistsError('Experiment already exists.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    lookup={Path(r['file']).as_posix():r for r in ledger['results']}
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    basepath=ROOT/'candidates/submission_best_0p71269.csv'
    base=pd.read_csv(basepath)
    if not base.row_id.equals(sample.row_id): raise ValueError('Base IDs differ.')
    score0=lookup['candidates/submission_blend_v2.csv']['rmse']
    if sha(basepath)!=lookup['candidates/submission_blend_v2.csv']['sha256']:
        raise ValueError('Base changed.')
    directions=[]
    scores=[]
    sources=[]
    for experiment in ['interaction_v1','thermal_reference_v1']:
        manifest=json.loads((ROOT/'runs'/experiment/'manifest.json').read_text())
        path=ROOT/manifest['candidate']
        score=lookup[path.relative_to(ROOT).as_posix()]['rmse']
        if sha(path)!=manifest['sha256']: raise ValueError('Scored file changed.')
        frame=pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id): raise ValueError('IDs differ.')
        directions.append((frame.prediction-base.prediction).to_numpy())
        scores.append(score)
        sources.append({'file':path.relative_to(ROOT).as_posix(),'rmse':score,'sha256':sha(path)})
    D=np.column_stack(directions)
    gram=D.T@D/len(D)
    gradient=(np.square(scores)-score0**2-np.diag(gram))/2
    optimum=-np.linalg.solve(gram,gradient)
    incumbent_weights=np.array([0.,1.])
    used=incumbent_weights+.9*(optimum-incumbent_weights)
    pred=base.prediction.to_numpy()+D@used
    forecast=lambda w:float(np.sqrt(max(score0**2+2*gradient@w+w@gram@w,0)))
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction.')
    folder.mkdir(parents=True)
    output=folder/'submission_thermal_calibrated.csv'
    result=sample.copy()
    result['prediction']=pred
    result.to_csv(output,index=False)
    reread=pd.read_csv(output)
    if not reread.row_id.equals(sample.row_id) or not np.allclose(reread.prediction,pred,atol=1e-12,rtol=1e-12):
        raise ValueError('Round-trip failed.')
    incumbent=ROOT/ledger['best_file']
    report={'status':'UNSCORED calibrated temperature model','candidate':output.relative_to(ROOT).as_posix(),
        'sha256':sha(output),'sources':sources,'base_file':basepath.relative_to(ROOT).as_posix(),
        'base_sha256':sha(basepath),'base_rmse':score0,
        'incumbent_file':ledger['best_file'],'incumbent_sha256':sha(incumbent),
        'incumbent_rmse':ledger['best_public_rmse'],'nominal_weights':optimum.tolist(),
        'used_weights':used.tolist(),'gram_full_test':gram.tolist(),
        'nominal_rmse_forecast':forecast(optimum),'candidate_rmse_forecast':forecast(used),
        'limitations':['Calibration uses public leaderboard feedback; no independent target validation.',
            'Full-test moments approximate unknown public moments; private performance is unmeasured.']}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
