"""Antithetic Sobol integration for the unchanged Gaussian reconstruction.

Nodes are deterministic for a given power/seed and reused by row, so predictions
do not depend on batch size or row ordering. Clipping matches the legacy cache.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from model3 import NL,derived
from reconstruction_repair import LOW,HIGH


def normal_nodes(power=8,seed=104):
    if power<2:raise ValueError('At least four nodes are required.')
    unit=qmc.Sobol(d=NL,scramble=True,seed=seed).random_base2(power-1)
    half=ndtri(np.clip(unit,1e-12,1-1e-12))
    return np.concatenate([half,-half])


def moments(th,cov,p,nodes,accepted=None):
    n=len(th)
    names=['pg','tau','eta','tl','S','D']
    out={prefix+k:np.zeros(n) for prefix in ['E_','V_'] for k in names}
    out.update({f'E_cap{c}':np.zeros(n) for c in [16,18,20,22]})
    out.update({'E_pm':np.zeros(n),'pg_panel':np.zeros(n),'panel':np.zeros(n),'solar':np.zeros(n)})
    L=np.linalg.cholesky(cov+np.eye(NL)*1e-12)
    for node in nodes:
        ts=th+np.einsum('nij,j->ni',L,node)
        ts[:,0]=np.clip(ts[:,0],0,1300)
        ts[:,3]=np.clip(ts[:,3],0,25)
        ts[:,5]=np.clip(ts[:,5],0,1.05)
        ts[:,6]=np.clip(ts[:,6],.005,1.05)
        if accepted is not None:ts[accepted]=np.clip(ts[accepted],LOW,HIGH)
        _,panel,tau,eta,solar,_,pg,tl,_=derived(ts,p)
        for key,value in dict(pg=pg,tau=tau,eta=eta,tl=tl,S=ts[:,6],D=ts[:,7]).items():
            out['E_'+key]+=value/len(nodes)
            out['V_'+key]+=value*value/len(nodes)
        pm=pg*tau*eta*(1-tl)
        out['E_pm']+=pm/len(nodes)
        for c in [16,18,20,22]:out[f'E_cap{c}']+=np.minimum(c,pm)/len(nodes)
        out['pg_panel']+=pg*(panel-25)/len(nodes)
        out['panel']+=panel/len(nodes)
        out['solar']+=solar/len(nodes)
    for key in names:out['V_'+key]=np.maximum(out['V_'+key]-out['E_'+key]**2,0)
    if any(not np.isfinite(value).all() for value in out.values()):raise ValueError('Nonfinite moments.')
    return out
