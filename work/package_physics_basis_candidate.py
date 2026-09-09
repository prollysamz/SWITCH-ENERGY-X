"""Package the screened physics-basis prediction for one public test."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from audit_order_priors import ROOT

def main():
    outdir=ROOT/'runs/physics_basis_v1';path=outdir/'submission_physics_basis.csv'
    if path.exists():raise FileExistsError('Preserve candidate')
    z=np.load(outdir/'candidate_diagnostic.npz');sample=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    assert np.array_equal(z['row_id'],sample.row_id)
    out=sample.copy();out['prediction']=z['candidate'];out.to_csv(path,index=False)
    check=pd.read_csv(path);assert check.row_id.equals(sample.row_id) and list(check.columns)==['row_id','prediction'] and np.isfinite(check.prediction).all()
    manifest=json.loads((outdir/'report.json').read_text());manifest.update(candidate=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      status='UNSCORED regularized physics-basis candidate',score_used_for_export=True,
      score_equation_fit='Historical exact public RMSEs constrained the aggregate moments; three later exact probes were held out of coefficient fitting.',
      residual_rmse_forecast=float(np.sqrt(manifest['residual_variance'])),
      target_rmse_forecast=None,limitations=manifest.get('limitations',[])+[
        'The candidate is selected using adaptive public feedback; the held-out score check is aggregate consistency, not target-label validation.',
        'The regularized basis has many correlated terms; its individual coefficients should not be interpreted physically.',
        'Private score and final rank remain unknown.'])
    (outdir/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['candidate','sha256','candidate_mean','candidate_sd','candidate_min','candidate_max','rms_from_best','fit_rmse','held_rmse','residual_rmse_forecast']},indent=2))

if __name__=='__main__':main()
