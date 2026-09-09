"""Held-out sensor validation. These errors are NOT target NDEM RMSE."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from model3 import Params, build_Y, COL, derived, map_latents
from model4 import map_latents as improved_map


def anchored_pg(df, blocks, prior_wind_per_density):
    """Use direct generation evidence before the nonlinear wind inversion."""
    pg = np.asarray(blocks['E_pg']).copy()
    f = lambda i: df[f'feature_{i:02d}'].to_numpy()
    # No surviving wind indicator: integrate the population wind distribution.
    no_wind = np.all(~np.isfinite(np.column_stack([f(i) for i in [4, 13, 14, 18, 21]])), axis=1)
    rho = np.where(np.isfinite(f(10)), f(10), 1.176)
    solar = np.where(np.isfinite(f(12)), f(12), 0.43)
    pg[no_wind] = (rho * prior_wind_per_density + solar)[no_wind]
    # For observed wind generation, the small solar contribution is easier to infer.
    wind = np.isfinite(f(13))
    pg[wind] = (f(13) + solar)[wind]
    # Grid frequency identifies gross generation plus demand.
    freq = np.isfinite(f(21))
    dem = np.where(np.isfinite(f(9)), f(9), blocks['E_D'])
    pg[freq] = ((f(21)-50.0)/0.9 + dem)[freq]
    gross = np.isfinite(f(14))
    pg[gross] = f(14)[gross]
    return pg


def main():
    p = pickle.load(open('work/params_final.pkl', 'rb'))
    train = pd.read_csv('Dataset/train.csv', usecols=['feature_10', 'feature_13'])
    prior = float((train.feature_13 / train.feature_10).mean())
    df = pd.read_csv('Dataset/test.csv')
    df = df[df.feature_14.notna()].sample(8000, random_state=908).reset_index(drop=True)
    truth = df.feature_14.to_numpy()
    scenarios = [[14], [14,21], [14,21,13], [14,21,13,18], [14,21,13,18,4]]
    results = []
    for hide in scenarios:
        masked = df.copy()
        for i in hide:
            masked[f'feature_{i:02d}'] = np.nan
        th, _ = map_latents(build_Y(masked), p, n_iter=22)
        pg = derived(th,p)[6]
        anchored = anchored_pg(masked, {'E_pg': pg, 'E_D': th[:,7]}, prior)
        improved_th, _ = improved_map(build_Y(masked), p, n_iter=22)
        improved = derived(improved_th,p)[6]
        for name, pred in [('original_map',pg), ('anchored',anchored), ('improved_map',improved)]:
            e = pred-truth
            rec = {'hidden':hide, 'method':name, 'n':len(e), 'rmse':float(np.sqrt(np.mean(e*e))),
                   'mae':float(np.abs(e).mean()), 'bias':float(e.mean()), 'max_abs':float(np.abs(e).max())}
            results.append(rec)
            print(json.dumps(rec), flush=True)
    Path('work/reconstruction_audit.json').write_text(json.dumps({'note':'Sensor reconstruction, not target validation.',
        'prior_wind_per_density':prior, 'results':results}, indent=2))


if __name__ == '__main__':
    main()
