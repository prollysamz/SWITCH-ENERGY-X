"""Compute the exact public-optimal step along the failed physics direction.

This is a diagnostic only; it uses the newly measured score and does not assume
the direction's large original amplitude was appropriate.
"""
import json,hashlib
import numpy as np,pandas as pd
from pathlib import Path
from audit_order_priors import ROOT

OUT=ROOT/'runs/physics_direction_opt_v1'

def main():
    if OUT.exists():raise FileExistsError('Preserve run')
    b=pd.read_csv(ROOT/'candidates/submission_best_0p43681.csv');p=pd.read_csv(ROOT/'runs/physics_basis_v1/submission_physics_basis.csv');s=pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    assert b.row_id.equals(p.row_id) and b.row_id.equals(s.row_id)
    d=p.prediction.to_numpy()-b.prediction.to_numpy();r0=.43681;r1=.58194
    q=float(np.mean(d*d)); inner=(q+r0*r0-r1*r1)/2.; step=inner/q
    pred=float(np.sqrt(max(r0*r0-inner*inner/q,0)))
    out=s.copy();out['prediction']=b.prediction.to_numpy()+step*d;OUT.mkdir();path=OUT/'submission_physics_optimal_step.csv';out.to_csv(path,index=False)
    rec=dict(status='UNSCORED exact optimum along measured physics-basis direction',candidate=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      incumbent_file='candidates/submission_best_0p43681.csv',incumbent_rmse=r0,direction_file='runs/physics_basis_v1/submission_physics_basis.csv',direction_rmse=r1,
      direction_rms=float(np.sqrt(q)),measured_inner_product=inner,cosine_to_incumbent_residual=float(inner/(np.sqrt(q)*r0)),optimal_step=step,conditional_public_rmse=pred,
      prediction_rms_change=float(abs(step)*np.sqrt(q)),target_rmse_forecast=None,
      decision='Do not spend a submission: the measured gain is only about 0.0023 and cannot approach P1.',
      limitation='The public identity is exact up to rounded reported scores; private gain is unmeasured.')
    (OUT/'manifest.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))

if __name__=='__main__':main()
