"""Export the frozen, independently audited learned correction."""
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from learned_energy_distillation import ROOT, OUT, COLS
from confirm_learned_energy import regimes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = OUT/'submission_learned_energy.csv'
    if output.exists():
        raise FileExistsError('Preserve candidate')
    validation = json.loads((OUT/'confirmation.json').read_text())
    if not validation['gate']:
        raise ValueError('Two weighted audits must pass')
    if sha(OUT/'model.pkl') != validation['model_sha256']:
        raise ValueError('Audited model changed')
    state = pickle.loads((OUT/'model.pkl').read_bytes())
    test = pd.read_csv(ROOT/'Dataset/test.csv')
    sample = pd.read_csv(ROOT/'Dataset/sample_submission.csv')
    baseline_path = ROOT/'candidates/submission_best_0p60058.csv'
    baseline = pd.read_csv(baseline_path)
    if sha(baseline_path) != 'ef2cba635975b56601a4acfe4d4a5814afa6e7f4cb8941cf9ef6170bee530f36':
        raise ValueError('Preserved best changed')
    cache = np.load(ROOT/'runs/deterministic_v1/posterior_repaired.npz')
    for ids in [sample.row_id, baseline.row_id, cache['row_id']]:
        if not np.array_equal(ids, test.row_id):
            raise ValueError('IDs differ')
    raw = test[COLS].to_numpy()
    extras = np.column_stack([baseline.prediction, cache['mean'],
        np.sqrt(np.maximum(np.diagonal(cache['covariance'], axis1=1, axis2=2), 0))])
    inputs = np.column_stack([raw, extras])
    group = regimes(raw)
    weight = np.array([state['weights'][g] for g in group])
    change = weight*state['model'].predict(inputs)
    # Inference must not depend on batching or row ordering.
    reverse = state['model'].predict(inputs[::-1])[::-1]
    np.testing.assert_array_equal(change, weight*reverse)
    frame = sample.copy()
    frame['prediction'] = baseline.prediction.to_numpy() + change
    if not frame.row_id.is_unique or not np.isfinite(frame.prediction).all():
        raise ValueError('Invalid output')
    frame.to_csv(output, index=False)
    check = pd.read_csv(output)
    if list(check.columns) != ['row_id', 'prediction'] or not check.row_id.equals(test.row_id):
        raise ValueError('Round-trip schema failed')
    np.testing.assert_allclose(check.prediction, frame.prediction, atol=1e-12, rtol=1e-12)
    source = ['learned_sensor_benchmark.py', 'learned_energy_distillation.py',
              'confirm_learned_energy.py', 'build_learned_energy_submission.py']
    report = dict(status='Unscored candidate from self-supervised learned corrections.',
        candidate=output.relative_to(ROOT).as_posix(), sha256=sha(output),
        incumbent_file=baseline_path.relative_to(ROOT).as_posix(), incumbent_sha256=sha(baseline_path),
        incumbent_rmse=.60058, target_rmse_forecast=None, allow_score_calibration=False,
        validation=validation, n_rows=len(frame), calibration_weights=state['weights'],
        rms_prediction_change=float(np.sqrt(np.mean(change**2))), mean_change=float(change.mean()),
        maximum_absolute_change=float(np.max(np.abs(change))),
        changes_by_regime={g:dict(n=int(np.sum(group == g)), rms=float(np.sqrt(np.mean(change[group == g]**2))))
                          for g in ['wind', 'thermal', 'other']},
        code_sha256={f:sha(ROOT/'work'/f) for f in source},
        limitations=['No hidden target labels used in these audits.',
            'The frozen teacher formula remains a source of possible error.',
            'Observed gains primarily concern rare information-poor wind rows.',
            'Natural missingness differs from artificial masking; the public score remains unmeasured.'])
    (OUT/'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['validation', 'code_sha256']}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
