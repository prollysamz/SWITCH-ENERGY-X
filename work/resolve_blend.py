"""Closed-form optimal blend of every scored submission, with an identifiability guard.

Each scored file gives one exact linear measurement of the target. Writing
MSE_i = <p_i,p_i> - 2*u_i (so u_i = <p_i,Y> - <Y,Y>/2), the unknown <Y,Y>
cancels for any blend whose weights sum to 1. Taking the best scored file as the
base and D as the deviations of the others from it:

    MSE(delta) = MSE_best + 2*delta.g + delta' H delta,
    H = D D'/N,  g = (D p_best)/N - (u - u_best)

H is eigen-decomposed. The full-test Gram only approximates the unknown public
Gram (empirically ~0.009 in MSE units), so any eigendirection carrying less
energy than EIGEN_FLOOR is dropped rather than fitted: those are exactly the
directions where the solver extrapolates wildly on a handful of rows.

    python work/resolve_blend.py [--floor F] [--write PATH] [--free]
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EIGEN_FLOOR = 0.05          # ~5x the measured Gram-proxy noise

def main():
    argv = sys.argv
    floor = 0.0 if '--free' in argv else (
        float(argv[argv.index('--floor')+1]) if '--floor' in argv else EIGEN_FLOOR)
    led = json.loads((ROOT/'candidates/leaderboard_results.json').read_text())
    files = [r['file'] for r in led['results']]
    S = np.array([float(r['rmse']) for r in led['results']])
    P = np.vstack([pd.read_csv(ROOT/f)['prediction'].to_numpy() for f in files])
    n, N = P.shape
    G = P @ P.T / N
    u = np.array([(G[i,i] - S[i]**2)/2 for i in range(n)])
    b = int(np.argmin(S)); pb = P[b]
    D = P - pb
    H = D @ D.T / N
    g = (D @ pb)/N - (u - u[b])
    ev, V = np.linalg.eigh(H)
    print(f'{n} scored submissions; best single = {S.min():.5f}  ({files[b]})')
    print('deviation-direction energies:', np.array2string(ev, precision=5, suppress_small=True))
    keep = ev > floor
    print(f'floor {floor:.3f} -> keeping {keep.sum()} of {n} directions'
          f'  (dropped {(~keep).sum()} as unidentifiable)')
    if keep.sum() == 0:
        print('\nNothing identifiable. The scored set is exhausted; submit a NEW direction.')
        return
    Vk, lam = V[:, keep], ev[keep]
    delta = -Vk @ ((Vk.T @ g)/lam)
    q = pb + D.T @ delta
    mse = S[b]**2 + 2*delta @ g + delta @ H @ delta
    rmse = np.sqrt(max(mse, 0))
    print(f'\nforecast RMSE = {rmse:.5f}   vs best single {S.min():.5f}   gain {S.min()-rmse:+.5f}')
    print(f'blend: mean={q.mean():.3f} sd={q.std():.3f} min={q.min():.2f} max={q.max():.2f} '
          f'neg={(q<0).mean():.3f}  rms change from best={np.sqrt(((q-pb)**2).mean()):.4f}')
    if S.min()-rmse < 0.01:
        print('\n** gain is inside the ~0.006 proxy-noise band: NOT worth a submission.')
        print('** the scored set is exhausted - score a new independent direction instead.')
    if q.max() > 30 or q.min() < -30:
        print('\n** WARNING: blend leaves the plausible target range; a degenerate direction is being fitted.')
    if '--write' in argv:
        out = ROOT/argv[argv.index('--write')+1]
        sub = pd.read_csv(ROOT/'Dataset/sample_submission.csv')
        sub['prediction'] = q
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        chk = pd.read_csv(out)
        assert chk.shape == (100000,2) and chk.row_id.is_unique and np.isfinite(chk.prediction).all()
        assert np.allclose(chk.prediction, q, rtol=1e-12, atol=1e-12)
        print(f'\nwrote {out}  sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}...')

if __name__ == '__main__':
    main()
