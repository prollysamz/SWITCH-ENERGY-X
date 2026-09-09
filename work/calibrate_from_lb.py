"""Approximate rescaling from two aggregate scores, if competition rules permit.

Both scores must use identical evaluation rows. Without their IDs, using full-test
prediction moments is approximate. This does not identify the target mean,
correlation, hidden formula, or private score.
Usage: python work/calibrate_from_lb.py RMSE_constant RMSE_baseline [constant]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def estimate_scale(prediction, rmse_constant, rmse_model, constant=3.9):
    y=np.asarray(prediction,dtype=float)
    scores=np.array([rmse_constant,rmse_model,constant],dtype=float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(scores).all() or np.any(scores[:2]<0):
        raise ValueError('Predictions, constant and nonnegative scores must be finite.')
    direction=y-constant
    second_moment=float(np.mean(direction**2))
    if second_moment<=1e-12:
        raise ValueError('Baseline is indistinguishable from the constant.')
    cross_moment=(rmse_constant**2+second_moment-rmse_model**2)/2
    residual=rmse_constant**2-cross_moment**2/second_moment
    if residual < -1e-4:
        raise ValueError('Scores and prediction moments are inconsistent; check files and evaluation split.')
    factor=cross_moment/second_moment
    return constant+factor*direction, factor, float(np.sqrt(max(residual,0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rmse_constant',type=float)
    parser.add_argument('rmse_model',type=float)
    parser.add_argument('constant',type=float,nargs='?',default=3.9)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sub=pd.read_csv(root/'submission.csv')
    pred,factor,estimate=estimate_scale(sub.prediction,args.rmse_constant,args.rmse_model,args.constant)
    sub['prediction']=pred
    path=root/'submission_rescaled_approx.csv'
    sub.to_csv(path,index=False)
    print(f'Approximate factor {factor:.5f}; same-distribution RMSE estimate {estimate:.5f}')
    print('Public subset moments are unavailable. This is not a guaranteed optimum or private-score forecast.')
    print(f'Wrote {path}')


if __name__=='__main__':
    main()
