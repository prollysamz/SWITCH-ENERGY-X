"""Test whether observed panel-temperature deviations carry target signal.

If the target was generated from noisy feature values rather than clean latent
states, denoising every observed feature can discard useful signal. This probe
tests that distinction for the temperature sensor, without assuming the answer.
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'runs/observed_thermal_v1'
    if folder.exists(): raise FileExistsError('Experiment already exists.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    test=pd.read_csv(ROOT/'Dataset/test.csv',usecols=['row_id','feature_07'])
    incumbent=pd.read_csv(ROOT/ledger['best_file'])
    blocks=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    params=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    if not np.array_equal(blocks['row_id'],sample.row_id) or not incumbent.row_id.equals(sample.row_id) or not test.row_id.equals(sample.row_id):
        raise ValueError('Input IDs differ.')
    mean=(1-blocks['E_tau'])/params.beta+params.Tref
    sd=np.maximum(np.sqrt(blocks['V_tau'])/params.beta,1e-9)
    delta=mean-25
    a=delta/sd
    reconstructed=(sd*sd+delta*delta)*ndtr(a)+delta*sd*np.exp(-a*a/2)/np.sqrt(2*np.pi)
    observed=test.feature_07.notna().to_numpy()
    raw=np.where(observed,-(np.maximum(test.feature_07.to_numpy()-25,0)**2-reconstructed),0.)
    scored=[]
    sources=[]
    for row in ledger['results']:
        path=ROOT/row['file']
        digest=sha(path)
        if 'sha256' in row and digest!=row['sha256']: raise ValueError('Source changed.')
        frame=pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id): raise ValueError('Source IDs differ.')
        scored.append(frame.prediction.to_numpy())
        sources.append({'file':row['file'],'sha256':digest})
    effect,projection=orthogonal_effect(raw,np.column_stack(scored))
    change=.25*effect
    result=sample.copy()
    result['prediction']=incumbent.prediction.to_numpy()+change
    if not np.isfinite(result.prediction).all(): raise ValueError('Nonfinite prediction.')
    folder.mkdir(parents=True)
    path=folder/'submission_observed_temperature.csv'
    result.to_csv(path,index=False)
    np.save(folder/'direction.npy',change)
    reread=pd.read_csv(path)
    if not reread.row_id.equals(sample.row_id) or not np.allclose(reread.prediction,result.prediction,atol=1e-12,rtol=1e-12):
        raise ValueError('Round-trip failed.')
    report={'status':'UNSCORED observed-sensor hypothesis','candidate':path.relative_to(ROOT).as_posix(),
        'sha256':sha(path),'incumbent_file':ledger['best_file'],'incumbent_rmse':ledger['best_public_rmse'],
        'incumbent_sha256':sha(ROOT/ledger['best_file']),'sources':sources,
        'hypothesis':'Observed panel-temperature deviations affect the hidden target beyond the reconstructed latent state.',
        'raw_effect':'For observed feature_07 only: -(max(feature_07-25,0)^2 - E[max(panel-25,0)^2]); zero otherwise.',
        'effect_sd':.25,'projection':projection,'observed_fraction':float(observed.mean()),
        'rms_change':float(np.sqrt(np.mean(change*change))),
        'max_abs_change':float(abs(change).max()),
        'limitations':['The target generation order is unknown; this hypothesis requires a measured score.',
            'Full-test orthogonalization is not independent target validation.']}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
