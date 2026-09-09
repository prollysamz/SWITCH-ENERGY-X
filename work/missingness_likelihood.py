"""Likelihood of the natural sensor mask under the fitted failure model."""
import numpy as np
from scipy.special import expit


def failure_weights(th,mask,params,model):
    panel=th[:,1]+params.a1*th[:,0]-params.a2*th[:,3]
    weights=np.ones(len(th))
    for group,values in [('temperature',panel),('humidity',th[:,2])]:
        spec=model[group]
        q=spec['coefficients']
        p=np.clip(q[0]+q[1]*expit((values-q[2])/q[3]),1e-9,1-1e-9)
        count=sum(mask[:,i-1] for i in spec['features'])
        n=len(spec['features'])
        weights*=p**count*(1-p)**(n-count)
    return weights
