"""Test the missing linear temperature term beyond scored thermal curves.

A quadratic centered at a different reference temperature requires a linear
term as well as the existing quadratic. Include colder conditions explicitly.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from build_interaction_candidate import orthogonal_effect

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'runs/temperature_slope_v1'
    if folder.exists():raise FileExistsError('Experiment already exists.')
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    incumbent=pd.read_csv(ROOT/ledger['best_file'])
    blocks=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    params=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    if not np.array_equal(blocks['row_id'],sample.row_id) or not incumbent.row_id.equals(sample.row_id):
        raise ValueError('Input IDs differ.')
    temperature=(1-blocks['E_tau'])/params.beta+params.Tref
    scored=[]
    sources=[]
    for row in ledger['results']:
        path=ROOT/row['file']
        digest=sha(path)
        if 'sha256' in row and digest!=row['sha256']:raise ValueError('Source changed.')
        frame=pd.read_csv(path)
        if not frame.row_id.equals(sample.row_id):raise ValueError('Source IDs differ.')
        scored.append(frame.prediction.to_numpy())
        sources.append({'file':row['file'],'sha256':digest})
    effect,projection=orthogonal_effect(-temperature,np.column_stack(scored))
    change=.5*effect
    result=sample.copy()
    result['prediction']=incumbent.prediction.to_numpy()+change
    if not np.isfinite(result.prediction).all():raise ValueError('Nonfinite prediction.')
    folder.mkdir(parents=True)
    path=folder/'submission_temperature_slope.csv'
    result.to_csv(path,index=False)
    np.save(folder/'direction.npy',change)
    reread=pd.read_csv(path)
    if not reread.row_id.equals(sample.row_id) or not np.allclose(reread.prediction,result.prediction,atol=1e-12,rtol=1e-12):
        raise ValueError('Round-trip failed.')
    report={'status':'UNSCORED temperature-slope hypothesis','candidate':path.relative_to(ROOT).as_posix(),
        'sha256':sha(path),'incumbent_file':ledger['best_file'],'incumbent_rmse':ledger['best_public_rmse'],
        'incumbent_sha256':sha(ROOT/ledger['best_file']),'sources':sources,
        'hypothesis':'An independent linear temperature term corrects the reference and cold-side behavior of the thermal curve.',
        'raw_effect':'-E[panel_temperature]','effect_sd':.5,'projection':projection,
        'rms_change':float(np.sqrt(np.mean(change*change))),'max_abs_change':float(abs(change).max()),
        'limitations':['Sign and strength require score feedback.','Full-test projection moments approximate public moments.']}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
