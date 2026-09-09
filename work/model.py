"""Structural generative model of the SWITCH ENERGY-X system + MAP latent reconstruction."""
import numpy as np

LAT = ['G','T','H','V','P','C','S','D']   # irradiance, temp, humidity, wind, pressure, cloud, soc, demand
NL = len(LAT)

P0 = dict(a1=0.021952, a2=0.344874, beta=0.004181, Tref=25.0,
          e0=0.913869, e1=0.045083,                 # eta = e0 + e1*sqrt(S)
          K=0.9569, ca=0.4099, cb=0.2979,           # solar
          kw=0.014448, Prated=21.32,                # wind
          l0=0.02997, l1=0.112093, lcap=0.079656,   # transmission loss
          d0=0.217357, d1=0.016654, dsd=0.1194,     # demand | temp
          fq=0.9)

def derived(th, p):
    G,T,H,V,Pr_,C,S,D = [th[...,i] for i in range(NL)]
    rho  = Pr_*100.0/(287.05*(T+273.15))
    Tp   = T + p['a1']*G - p['a2']*V
    tau  = 1.0 - p['beta']*(Tp - p['Tref'])
    eta  = p['e0'] + p['e1']*np.sqrt(np.clip(S,1e-9,None))
    psol = p['K']*(G/1000.0)*tau*eta*(1.0-p['ca']*C)*(1.0-p['cb']*H/100.0)
    pwin = np.minimum(p['Prated'], p['kw']*rho*np.clip(V,0,None)**3)
    pg   = psol + pwin
    tl   = np.minimum(p['lcap'], p['l0'] + p['l1']*psol)
    freq = 50.0 + p['fq']*(pg - D)
    return dict(rho=rho, Tp=Tp, tau=tau, eta=eta, psol=psol, pwin=pwin, pg=pg, tl=tl, freq=freq)

def forward(th, p):
    """-> (n,21) predicted observables for feature ids in OBS_IDS order."""
    G,T,H,V,Pr_,C,S,D = [th[...,i] for i in range(NL)]
    d = derived(th, p)
    return np.stack([G, T, H, V, Pr_, C, d['Tp'], S, D, d['rho'], d['eta'], d['psol'],
                     d['pwin'], d['pg'], d['tau'], G*H/100.0, V*V, S*D, d['freq'], d['tl']], axis=-1)

OBS_IDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,21,22]   # 20 structural observables
NO = len(OBS_IDS)
PRIOR_MU = np.array([718.5, 26.99, 66.89, 5.328, 1013.0, 0.3547, 0.5538, 0.6669])
PRIOR_SD = np.array([336.9,  7.001, 16.91, 2.792,   12.09, 0.1798, 0.1752, 0.1668])

def map_latents(Y, Wobs, p, prior_mu, prior_sd, n_iter=25, lam0=1e-3, verbose=False):
    """Y:(n,NO) observations with NaN, Wobs:(NO,) precision weights (1/sigma).
       Returns MAP latents (n,NL) and per-row posterior covariance (n,NL,NL)."""
    n = Y.shape[0]
    obs = ~np.isnan(Y)
    Yf = np.nan_to_num(Y, nan=0.0)
    th = np.tile(prior_mu, (n,1)).astype(np.float64)
    # warm start from directly-observed features
    for j,fid in enumerate(OBS_IDS):
        if fid in (1,2,3,4,5,6,8,9):
            k = {1:0,2:1,3:2,4:3,5:4,6:5,8:6,9:7}[fid]
            th[obs[:,j],k] = Y[obs[:,j],j]
    sc = prior_sd.copy()
    lam = np.full(n, lam0)
    def resid(t):
        pred = forward(t, p)
        r = (pred - Yf)*Wobs*obs                      # (n,NO)
        rp = (t - prior_mu)/prior_sd                  # (n,NL) prior
        rd = ((t[:,7] - (p['d0']+p['d1']*t[:,1]))/p['dsd'])[:,None]   # demand|temp
        rp = rp.copy(); rp[:,7] = 0.0                 # D handled by conditional prior
        return np.concatenate([r, rp, rd], axis=1)
    r = resid(th); cost = (r*r).sum(1)
    for it in range(n_iter):
        NR = r.shape[1]
        J = np.empty((n, NR, NL))
        for k in range(NL):
            h = 1e-5*sc[k]
            tp = th.copy(); tp[:,k] += h
            J[:,:,k] = (resid(tp) - r)/h
        JT = np.swapaxes(J,1,2)
        A = JT @ J
        g = (JT @ r[:,:,None])[:,:,0]
        idx = np.arange(NL)
        ok = np.zeros(n, bool)
        for _try in range(6):
            Ad = A.copy(); Ad[:,idx,idx] *= (1.0+lam[:,None])
            Ad[:,idx,idx] += 1e-12
            try: step = np.linalg.solve(Ad, -g[:,:,None])[:,:,0]
            except np.linalg.LinAlgError: break
            cand = th + step
            rc = resid(cand); cc = (rc*rc).sum(1)
            imp = (cc < cost) & ~ok
            th[imp] = cand[imp]; r[imp] = rc[imp]; cost[imp] = cc[imp]
            lam[imp] *= 0.3; ok |= imp
            lam[~ok] *= 8.0
            if ok.all(): break
        if verbose: print(f'   iter {it:2d} mean cost {cost.mean():.4f}')
    # posterior covariance (Gauss-Newton, unit-variance residuals)
    NR = r.shape[1]
    J = np.empty((n, NR, NL))
    for k in range(NL):
        h = 1e-5*sc[k]; tp = th.copy(); tp[:,k] += h
        J[:,:,k] = (resid(tp) - r)/h
    A = np.swapaxes(J,1,2) @ J
    A[:,np.arange(NL),np.arange(NL)] += 1e-9
    cov = np.linalg.inv(A)
    return th, cov

def build_Y(df):
    import numpy as np
    return np.column_stack([df[f'feature_{i:02d}'].values for i in OBS_IDS]).astype(np.float64)
