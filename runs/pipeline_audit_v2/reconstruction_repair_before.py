"""Versioned reconstruction repair; does not change the NDEM target formula.

Retains the old wind spline below 14. Uses a train-fitted smooth positive tail,
and retries implausible or inconsistent MAP solutions with bounded robust fits.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from model3 import (Params, COL, LAT, NL, OBS_IDS, SIG, PRIOR_MU, PRIOR_SD,
                    derived, forward, resid_fn, build_Y, map_latents as original_map)

LOW=np.array([0.,-20.,0.,0.,900.,0.,.01,0.])
HIGH=np.array([1400.,70.,100.,30.,1100.,1.,1.,2.])
RESIDUAL_LIMIT=8.0


class RepairedParams(Params):
    @classmethod
    def from_original(cls,p,tail):
        out=cls()
        out.__dict__.update(copy.deepcopy(p.__dict__))
        out.tail_coef=np.asarray(tail['coefficients'],dtype=float)
        return out

    def wind(self,V):
        v=np.maximum(np.asarray(V,dtype=float),0.)
        old=super().wind(v)
        c=self.tail_coef
        tail=np.exp(c[0])*v**3*expit(-(v-c[1])/c[2])
        # C1 transition avoids a value/slope discontinuity at the join.
        u=np.clip(v-14.,0.,1.)
        blend=u*u*(3-2*u)
        return (1-blend)*old+blend*tail


def diagnostics(th,Y,p):
    r=(forward(th,p)-Y)/SIG
    max_res=np.max(np.where(np.isfinite(r),abs(r),0),axis=1)
    outside=np.any((th<LOW)|(th>HIGH)|~np.isfinite(th),axis=1)
    return max_res,outside


def informed_start(Y,p):
    th=PRIOR_MU.copy()
    for fid,k in [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(8,6),(9,7)]:
        if np.isfinite(Y[COL[fid]]):
            th[k]=Y[COL[fid]]
    if np.isfinite(Y[COL[18]]):
        th[3]=np.sqrt(max(Y[COL[18]],0))
    if np.isfinite(Y[COL[7]]):
        panel=Y[COL[7]]
        if not np.isfinite(Y[COL[2]]):
            th[1]=panel-p.a1*th[0]+p.a2*th[3]
        elif not np.isfinite(Y[COL[1]]):
            th[0]=(panel-th[1]+p.a2*th[3])/p.a1
    elif np.isfinite(Y[COL[15]]) and not np.isfinite(Y[COL[2]]):
        panel=(1-Y[COL[15]])/p.beta+p.Tref
        th[1]=panel-p.a1*th[0]+p.a2*th[3]
    return np.clip(th,LOW+1e-8,HIGH-1e-8)


def robust_cost(r,scale=2.5):
    return float(np.sum(2*scale**2*(np.sqrt(1+(r/scale)**2)-1)))


def map_latents(Y,p,n_iter=22,need_cov=False,return_info=False):
    th,cov=original_map(Y,p,n_iter=n_iter,need_cov=need_cov)
    before,outside=diagnostics(th,Y,p)
    retry=outside|(before>RESIDUAL_LIMIT)
    accepted=np.zeros(len(Y),bool)
    for i in np.flatnonzero(retry):
        row=Y[i:i+1]
        observed=np.isfinite(row)
        filled=np.nan_to_num(row,nan=0.)
        def residual(x):
            return resid_fn(x[None,:],filled,observed,1/SIG,p)[0]
        starts=[informed_start(Y[i],p),np.clip(th[i],LOW+1e-8,HIGH-1e-8)]
        best=None
        for start in starts:
            fit=least_squares(residual,start,bounds=(LOW,HIGH),loss='soft_l1',
                f_scale=2.5,x_scale=PRIOR_SD,max_nfev=160,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            score=robust_cost(residual(fit.x))
            if np.isfinite(score) and (best is None or score<best[0]):
                best=(score,fit)
        if best is not None and (outside[i] or best[0]<robust_cost(residual(th[i]))):
            fit=best[1]
            th[i]=fit.x
            accepted[i]=True
            if need_cov:
                # scipy's robust-loss Jacobian downweights discrepant sensors.
                precision=fit.jac.T@fit.jac+np.diag(1e-8/PRIOR_SD**2)
                cov[i]=np.linalg.inv(precision)
    after,outside_after=diagnostics(th,Y,p)
    if np.any(outside_after):
        raise RuntimeError('Bounded repair left an invalid latent solution.')
    info={'retry':retry,'accepted':accepted,'max_residual_before':before,
          'max_residual_after':after,'still_inconsistent':after>RESIDUAL_LIMIT}
    if return_info:
        return th,cov,info
    return th,cov
