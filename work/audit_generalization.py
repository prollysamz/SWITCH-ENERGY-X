"""Read-only model diagnostics; writes an audit JSON, never a prediction CSV.

No local target labels exist. This audit cannot estimate private target RMSE.
"""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from model3 import build_Y, derived, map_latents
from build_hypotheses import components

ROOT=Path(__file__).resolve().parents[1]


def metrics(x):
    return {'rmse':float(np.sqrt(np.mean(x*x))), 'mae':float(np.mean(abs(x))),
            'bias':float(np.mean(x)), 'max_abs':float(np.max(abs(x)))}


def main():
    folder=ROOT/'candidates'
    ledger=json.loads((folder/'leaderboard_results.json').read_text())
    train=pd.read_csv(ROOT/'Dataset/train.csv')
    test=pd.read_csv(ROOT/'Dataset/test.csv')
    sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    tr=np.load(ROOT/'work/blocks_train.npz')
    te=np.load(ROOT/'work/blocks_test.npz')
    p=pickle.load(open(ROOT/'work/params_final.pkl','rb'))
    report={'scope':'Code/data audit only; no NDEM labels or independent target validation',
            'new_submission_generated':False,'integrity':[]}
    predictions={}
    for row in ledger['results']:
        path=ROOT/row['file']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==row['sha256']
        df=pd.read_csv(path)
        assert list(df.columns)==['row_id','prediction'] and len(df)==100000
        assert df.row_id.is_unique and np.array_equal(df.row_id,sample.row_id)
        assert np.isfinite(df.prediction).all()
        predictions[row['file']]=df.prediction.to_numpy()
        report['integrity'].append({'file':row['file'],'sha256_verified':True,'schema_ids_finite':True})
    b=predictions['candidates/submission_battery.csv']
    c=predictions['candidates/submission_combined.csv']
    o=predictions['submission.csv']
    best=predictions['candidates/submission_refined_plane.csv']
    D=np.column_stack([c-b,o-b])
    all_diff=np.column_stack([pred-b for pred in predictions.values()])
    singular=np.linalg.svd(all_diff,compute_uv=False)
    gram=D.T@D/len(D)
    report['global_tuning_capacity']={'affine_rank_of_five_scored_predictions':int(np.sum(singular>1e-8*singular[0])),
        'singular_values':singular.tolist(),'direction_gram_condition_number':float(np.linalg.cond(gram)),
        'note':'Rank is final global weight capacity, not total adaptive selection complexity.'}
    # Reconstruct the old ensemble from the input caches without rerunning writes.
    ctr,cte=components(tr),components(te)
    both=np.concatenate([ctr,cte],axis=1)
    standardized=(both-both.mean(axis=1)[:,None])/both.std(axis=1)[:,None]
    blend=np.array([.25,.20,.20,.15,.10,.10])@standardized
    rebuilt=3.9+4.4*(blend-blend.mean())/blend.std()
    report['baseline_rebuild_max_abs_difference']=float(np.max(abs(rebuilt[len(train):]-o)))
    manifest=json.loads((folder/'refined_plane_manifest.json').read_text())
    w=np.array(manifest['weights'])
    report['best_rebuild_max_abs_difference']=float(np.max(abs(b+D@w-best)))
    hypothesis=json.loads((folder/'manifest.json').read_text())
    comb_sd=hypothesis['normalization']['combined']['sd']
    battery_effect=2*(1-w.sum())+w[0]*2*.8/comb_sd
    heat_effect=w[0]*2*.6/comb_sd
    report['effective_effects']={
        'battery_standardized_coefficient':float(battery_effect),
        'heat_standardized_coefficient':float(heat_effect),
        'coefficient_on_raw_negative_drawdown':float(battery_effect/hypothesis['normalization']['battery']['sd']),
        'coefficient_on_raw_negative_heat_hinge':float(heat_effect/hypothesis['normalization']['heat']['sd']),
        'note':'These are conditional algebraic coefficients after centering and removing generation correlation.'}
    shift=[]
    for col in train.columns:
        if col=='row_id': continue
        pooled=np.sqrt((train[col].var()+test[col].var())/2)
        shift.append({'feature':col,'train_mean':float(train[col].mean()),'test_mean':float(test[col].mean()),
                      'standardized_mean_shift':float((test[col].mean()-train[col].mean())/pooled),
                      'missingness_difference':float(test[col].isna().mean()-train[col].isna().mean())})
    report['largest_train_test_shifts']=sorted(shift,key=lambda r:abs(r['standardized_mean_shift']),reverse=True)[:6]
    report['cache_has_row_ids']='row_id' in te.files
    report['cached_physical_anomalies']={}
    for name,z in [('train',tr),('test',te)]:
        tp=(1-z['E_tau'])/p.beta+p.Tref
        report['cached_physical_anomalies'][name]={
            'soc_above_one':int(np.sum(z['E_S']>1)), 'demand_below_zero':int(np.sum(z['E_D']<0)),
            'panel_temperature_below_minus40':int(np.sum(tp < -40)),
            'min_inferred_panel_temperature':float(tp.min()),'min_inferred_demand':float(z['E_D'].min())}
    report['fitted_vs_hardcoded_thermal_beta']={'fitted':float(p.beta),'hardcoded':.004213}
    # Jensen gap from using max(E[T]-40,0), rather than E[max(T-40,0)].
    # Gaussian marginal approximation; a diagnostic, not a replacement model.
    mean=(1-te['E_tau'])/.004213+25
    sd=np.maximum(np.sqrt(te['V_tau'])/.004213,1e-9)
    delta=mean-40
    a=delta/sd
    expected_hinge=sd*np.exp(-.5*a*a)/np.sqrt(2*np.pi)+delta*ndtr(a)
    hinge_gap=expected_hinge-np.maximum(delta,0)
    coefficient=report['effective_effects']['coefficient_on_raw_negative_heat_hinge']
    report['gaussian_heat_jensen_gap']={'heat_units_mean':float(hinge_gap.mean()),
        'prediction_units_rms':float(np.sqrt(np.mean((coefficient*hinge_gap)**2))),
        'prediction_units_max':float(np.max(coefficient*hinge_gap)),
        'note':'Magnitude of a potential approximation correction, not a predicted RMSE gain.'}
    influence=np.einsum('ni,ij,nj->n',D,np.linalg.inv(gram),D)
    report['direction_influence']={'top_one_percent_share':float(np.sort(influence)[-1000:].sum()/influence.sum()),
        'max_to_mean':float(influence.max()/influence.mean())}
    report['prediction_tails']={'below_minus20':int(np.sum(best < -20)),'above20':int(np.sum(best > 20)),
        'min':float(best.min()),'max':float(best.max()),'negative_fraction':float(np.mean(best<0))}
    # Fresh sensor holdout checks. Neither this nor synthetic unit tests validate NDEM.
    subset=test[test.feature_07.notna()].sample(5000,random_state=910).reset_index(drop=True)
    truth=subset.feature_07.to_numpy()
    sensor=[]
    for hidden in [[7],[7,15],[7,15,2]]:
        masked=subset.copy()
        for fid in hidden:
            masked[f'feature_{fid:02d}']=np.nan
        th,_=map_latents(build_Y(masked),p,n_iter=22)
        estimate=derived(th,p)[1]
        record={'hidden':hidden,'n':len(truth),**metrics(estimate-truth)}
        sensor.append(record)
        print('Sensor-only holdout:',json.dumps(record),flush=True)
    report['new_panel_temperature_holdout']=sensor
    report['forecast_vs_actual']=[{'predicted':1.0871456998767204,'observed':1.08381},
                                  {'predicted':.8066991046293289,'observed':.78886}]
    out=ROOT/'work/generalization_audit.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
