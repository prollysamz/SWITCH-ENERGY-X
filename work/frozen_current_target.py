"""Exact functional reconstruction of the scored 0.60061 predictor.

The regression below recovers algebraic coefficients from existing predictions,
not hidden target labels. It must reproduce an unused partition to machine
precision before the coefficients may be used with new reconstruction moments.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from build_hypotheses import components,expected_drawdown
from frozen_target import predict as repaired_predict

ROOT=Path(__file__).resolve().parents[1]
NAMES=['intercept','original','battery','combined','repaired',
       'repaired_heat40_power1p5','repaired_pg_panel','repaired_heat25_square']


def legacy_predictions(blocks,frozen):
    c=components(blocks)
    def blend(key):
        norm=frozen[key]
        v=np.asarray(frozen['component_weights'])@((c-np.asarray(norm['component_mean'])[:,None])/np.asarray(norm['component_sd'])[:,None])
        return (v-norm['blend_mean'])/norm['blend_sd']
    original=frozen['target_mean']+frozen['original_sd']*blend('original')
    g=blend('hypothesis')
    def residual(value,key):
        norm=frozen['normalization'][key]
        return (value-norm['mean']-norm['slope']*g)/norm['sd']
    bat=residual(-expected_drawdown(blocks),'battery')
    panel=(1-blocks['E_tau'])/frozen['heat_beta']+frozen['temperature_reference']
    heat=residual(-np.maximum(panel-frozen['heat_threshold'],0),'heat')
    comb=residual(.8*bat+.6*heat,'combined')
    base=frozen['target_mean']+frozen['generation_sd']*g
    return original,base+frozen['new_effect_sd']*bat,base+frozen['new_effect_sd']*comb


def matrix(old,repaired,pg_panel,frozen,beta,tref):
    original,battery,combined=legacy_predictions(old,frozen)
    panel=(1-repaired['E_tau'])/beta+tref
    sd=np.maximum(np.sqrt(repaired['V_tau'])/beta,1e-9)
    d=panel-25
    a=d/sd
    square=(sd*sd+d*d)*ndtr(a)+d*sd*np.exp(-a*a/2)/np.sqrt(2*np.pi)
    old_reference_panel=(1-repaired['E_tau'])/frozen['heat_beta']+frozen['temperature_reference']
    return np.column_stack([np.ones(len(panel)),original,battery,combined,
        repaired_predict(repaired,frozen),np.maximum(old_reference_panel-40,0)**1.5,
        pg_panel,square])


def predict(old,repaired,pg_panel,spec):
    X=matrix(old,repaired,pg_panel,spec['legacy_frozen'],spec['beta'],spec['tref'])
    return X@np.asarray(spec['coefficients'])


def main():
    folder=ROOT/'runs/deterministic_v1'
    folder.mkdir(exist_ok=True)
    output=folder/'frozen_target.json'
    if output.exists():raise FileExistsError('Preserve frozen target coefficients.')
    old=np.load(ROOT/'work/blocks_test.npz')
    rep=np.load(ROOT/'runs/reconstruction_v1/blocks_test.npz')
    joint=np.load(ROOT/'runs/interaction_v1/joint_features.npz')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    basepath=ROOT/'candidates/submission_best_0p60061.csv'
    base=pd.read_csv(basepath)
    for ids in [rep['row_id'],joint['row_id'],base.row_id]:
        if not np.array_equal(ids,sample.row_id):raise ValueError('IDs differ.')
    p=pickle.loads((ROOT/'work/params_final.pkl').read_bytes())
    frozen=json.loads((ROOT/'runs/reconstruction_v1/frozen_target.json').read_text())
    X=matrix(old,rep,joint['pg_panel'],frozen,p.beta,p.Tref)
    y=base.prediction.to_numpy()
    train=np.arange(len(y))%5!=0
    scale=X[train].std(axis=0);scale[0]=1
    b=np.linalg.lstsq(X[train]/scale,y[train],rcond=1e-12)[0]/scale
    error=X@b-y
    if abs(error).max()>1e-9:raise ValueError(f'Formula does not reproduce incumbent: {abs(error).max()}')
    spec={'status':'Frozen exact representation of scored predictions, not newly fitted target weights.',
        'base_file':basepath.relative_to(ROOT).as_posix(),'base_rmse':.60061,
        'base_sha256':hashlib.sha256(basepath.read_bytes()).hexdigest(),
        'names':NAMES,'coefficients':b.tolist(),'legacy_frozen':frozen,'beta':p.beta,'tref':p.Tref,
        'reproduction_max_abs_error':float(abs(error).max()),
        'unused_partition_max_abs_error':float(abs(error[~train]).max()),
        'standardized_design_condition_number':float(np.linalg.cond(X[train]/scale)),
        'limitations':['Exact reconstruction of predictions does not establish target accuracy.',
            'Legacy and repaired reconstruction branches remain distinct; no target coefficients or normalization are retuned.']}
    output.write_text(json.dumps(spec,indent=2))
    print(json.dumps({k:v for k,v in spec.items() if k!='legacy_frozen'},indent=2))


if __name__=='__main__':main()
