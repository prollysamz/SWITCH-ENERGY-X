"""Build the literal proxy-law candidate; no score is used to choose coefficients."""
import json,hashlib,pickle
import numpy as np,pandas as pd
from pathlib import Path
from reconstruction_repair import RepairedParams
from deterministic_moments import moments,normal_nodes
from model3 import build_Y
from learned_energy_distillation import stable_map
from audit_order_priors import ROOT

OUT=ROOT/'runs/direct_proxy_candidate_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve candidate')
    test=pd.read_csv(ROOT/'Dataset/test.csv');sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    original=pickle.loads((ROOT/'work/params_final.pkl').read_bytes()); p=RepairedParams.from_original(original,json.loads((ROOT/'work/wind_repair_v2.json').read_text()))
    prior=np.load(ROOT/'runs/order_prior_audit_v1/test_priors.npz')
    # Existing order-prior posteriors; local storage/demand priors are applied.
    post=np.load(ROOT/'runs/deterministic_v1/posterior_repaired.npz')
    mean,cov=post['mean'],post['covariance']
    from audit_order_priors import update
    mean,cov=update(mean,cov,prior['local'],prior['sd'],p)
    z=moments(mean,cov,p,normal_nodes(8,104),post['accepted'])
    def observed_or(fid,fallback):
        x=test[f'feature_{fid:02d}'].to_numpy();return np.where(np.isfinite(x),x,fallback)
    G=observed_or(14,z['E_pg']); tau=observed_or(15,z['E_tau']); eta=observed_or(11,z['E_eta'])
    D=observed_or(9,z['E_D']); S=observed_or(8,z['E_S']); tl=observed_or(22,z['E_tl'])
    raw=G*tau*eta*(1-tl)-D*(1-S)
    # Train full rows establish the proxy's natural center and scale, then use
    # the competition's approximate target moments only for a transparent affine
    # unit conversion (3.9, 4.6), never leaderboard feedback.
    train=pd.read_csv(ROOT/'Dataset/train.csv')
    complete=train.dropna(subset=[f'feature_{i:02d}' for i in [8,9,11,14,15,22]])
    traw=complete.feature_14*complete.feature_15*complete.feature_11*(1-complete.feature_22)-complete.feature_09*(1-complete.feature_08)
    center=float(traw.mean()); scale=float(traw.std()); prediction=3.9+(raw-center)*4.6/scale
    assert np.isfinite(prediction).all()
    out=sample.copy();out['prediction']=prediction;OUT.mkdir();path=OUT/'submission_direct_proxy.csv';out.to_csv(path,index=False)
    # Report independence from incumbent and score-derived span.
    ledger=json.loads((ROOT/'candidates/leaderboard_results.json').read_text());base=pd.read_csv(ROOT/ledger['best_file']).prediction.to_numpy()
    scored=[pd.read_csv(ROOT/r['file']).prediction.to_numpy() for r in ledger['results'] if r.get('precision')=='exact']
    span=np.column_stack([np.ones(len(raw))]+scored);direction=prediction-base;proj=span@np.linalg.lstsq(span,direction,rcond=1e-12)[0];novelty=float(np.mean((direction-proj)**2)/np.mean(direction**2))
    report=dict(status='UNSCORED literal feature-proxy target law',candidate=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      incumbent_file=ledger['best_file'],incumbent_rmse=ledger['best_public_rmse'],
      raw_formula='feature_14*feature_15*feature_11*(1-feature_22)-feature_09*(1-feature_08)',
      observed_proxy_fraction={str(i):float(test[f'feature_{i:02d}'].notna().mean()) for i in [8,9,11,14,15,22]},
      train_complete_rows=int(len(complete)),train_raw_center=center,train_raw_sd=scale,target_center_assumed=3.9,target_sd_assumed=4.6,
      test_prediction_mean=float(prediction.mean()),test_prediction_sd=float(prediction.std()),prediction_min=float(prediction.min()),prediction_max=float(prediction.max()),
      rms_from_incumbent=float(np.sqrt(np.mean(direction**2))),max_change=float(abs(direction).max()),scored_span_novelty=novelty,
      score_used_to_choose_formula=False,target_rmse_forecast=None,
      limitations=['The literal proxy law is motivated by the brief, not labels.',
        'Train complete rows have noisy proxies; affine scaling uses approximate brief moments.',
        'Missing values use the repaired posterior and local priors.',
        'No hidden-target or public-score validation exists.'])
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
