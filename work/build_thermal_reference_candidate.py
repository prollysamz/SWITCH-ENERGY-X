"""Test quadratic heat loss above the existing physical reference of 25 C.

Preserve the scored incumbent; integrate the squared positive part using the
repaired Gaussian temperature marginal, then remove all scored model directions.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from build_interaction_candidate import orthogonal_effect

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'runs/thermal_reference_v1'
    if folder.exists(): raise FileExistsError('Experiment already exists.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    incumbent=pd.read_csv(ROOT/ledger['best_file'])
    blocks=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    params=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    if not np.array_equal(blocks['row_id'],sample.row_id) or not incumbent.row_id.equals(sample.row_id):
        raise ValueError('Input row IDs mismatch.')
    mean=(1-blocks['E_tau'])/params.beta+params.Tref
    sd=np.maximum(np.sqrt(blocks['V_tau'])/params.beta,1e-9)
    delta=mean-25
    a=delta/sd
    expected_square=(sd*sd+delta*delta)*ndtr(a)+delta*sd*np.exp(-a*a/2)/np.sqrt(2*np.pi)
    scored=[]
    sources=[]
    for row in ledger['results']:
        path=ROOT/row['file']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if 'sha256' in row and digest!=row['sha256']: raise ValueError('Source changed.')
        frame=pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id): raise ValueError('Scored IDs mismatch.')
        scored.append(frame.prediction.to_numpy())
        sources.append({'file':row['file'],'sha256':digest})
    effect,projection=orthogonal_effect(-expected_square,np.column_stack(scored))
    change=.5*effect
    result=sample.copy()
    result['prediction']=incumbent.prediction.to_numpy()+change
    if not np.isfinite(result.prediction).all(): raise ValueError('Nonfinite predictions.')
    folder.mkdir(parents=True)
    path=folder/'submission_quadratic_heat25.csv'
    result.to_csv(path,index=False)
    np.save(folder/'direction.npy',change)
    reread=pd.read_csv(path)
    if not reread.row_id.equals(sample.row_id) or not np.allclose(reread.prediction,result.prediction,rtol=1e-12,atol=1e-12):
        raise ValueError('Output round-trip failed.')
    report={'status':'UNSCORED new temperature-shape hypothesis',
        'incumbent_file':ledger['best_file'],'incumbent_rmse':ledger['best_public_rmse'],
        'incumbent_sha256':hashlib.sha256((ROOT/ledger['best_file']).read_bytes()).hexdigest(),
        'candidate':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'hypothesis':'Additional squared heat penalty above the 25 C physical reference',
        'raw_effect':'-E[max(panel_temperature-25,0)^2] under repaired Gaussian marginal',
        'effect_sd':.5,'projection':projection,'sources':sources,
        'prediction_mean':float(result.prediction.mean()),'prediction_sd':float(result.prediction.std(ddof=0)),
        'prediction_min':float(result.prediction.min()),'prediction_max':float(result.prediction.max()),
        'limitations':['Unscored; no guaranteed improvement.','Gaussian marginal is approximate.',
                       'Full-test projection moments can differ from public moments.']}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
