"""Fit sensor-law parameters on unlabeled test rows for a holdout audit.

This never uses NDEM scores or target labels. It writes a separate parameter
file so the scored pipeline remains untouched.
"""
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from model3 import Params, build_Y, map_latents, derived, forward, COL, SIG, TW, bas, NL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "test_adaptation_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "params_test_adapted.pkl"
    if output.exists():
        raise FileExistsError("Preserve existing adaptation")
    frame = pd.read_csv(ROOT / "Dataset" / "test.csv")
    rng = np.random.default_rng(2209)
    ids = np.arange(len(frame))
    rng.shuffle(ids)
    fit_ids = ids[:15000]
    fit = frame.iloc[fit_ids].reset_index(drop=True)
    p0 = pickle.loads((ROOT / "work" / "params_final.pkl").read_bytes())
    p = copy.deepcopy(p0)
    Y = build_Y(fit)
    ob = ~np.isnan(Y)
    records = []
    for iteration in range(1):
        th, _ = map_latents(Y, p, n_iter=22)
        G, T, H, V, Pr, C, S, D = [th[:, i] for i in range(NL)]
        rho, Tp, tau, eta, psol, pwin, pg, tl, freq = derived(th, p)
        m = ob[:, COL[13]]
        if m.sum() > 100:
            B = bas(TW, np.clip(V[m], 0, 18)) * rho[m, None]
            d2 = np.diff(np.eye(B.shape[1]), 2, axis=0)
            lam = 0.01 * len(B) / B.shape[1]
            p.cw = np.linalg.solve(B.T @ B + lam * (d2.T @ d2), B.T @ Y[m, COL[13]])
        m = ob[:, COL[11]]
        if m.sum() > 100:
            p.e0, p.e1 = least_squares(
                lambda q: q[0] + q[1] * np.sqrt(np.clip(S[m], 1e-9, None)) - Y[m, COL[11]],
                [p.e0, p.e1], method="lm").x
        m = ob[:, COL[12]]
        if m.sum() > 100:
            p.K, p.ca, p.cb = least_squares(
                lambda q: q[0] * (G[m] / 1000) * tau[m] * eta[m] * (1 - q[1] * C[m]) *
                (1 - q[2] * H[m] / 100) - Y[m, COL[12]],
                [p.K, p.ca, p.cb], method="lm").x
        m7, m15 = ob[:, COL[7]], ob[:, COL[15]]
        def thermal(q):
            tp = T + q[0] * G - q[1] * V
            return np.concatenate([
                (tp[m7] - Y[m7, COL[7]]) / SIG[COL[7]],
                ((1 - q[2] * (tp[m15] - p.Tref)) - Y[m15, COL[15]]) / SIG[COL[15]],
            ])
        p.a1, p.a2, p.beta = least_squares(
            thermal, [p.a1, p.a2, p.beta], method="lm").x
        m = ob[:, COL[22]]
        if m.sum() > 100:
            p.l0, p.l1, p.lcap = least_squares(
                lambda q: np.minimum(q[2], q[0] + q[1] * psol[m]) - Y[m, COL[22]],
                [p.l0, p.l1, p.lcap], method="lm").x
        p.d0, p.d1 = np.linalg.lstsq(np.column_stack([np.ones(len(T)), T]), D, rcond=None)[0]
        p.dsd = max(float(np.std(D - (p.d0 + p.d1 * T))), 0.02)
        m = ob[:, COL[21]]
        if m.sum() > 100:
            p.f0, p.fq = np.linalg.lstsq(
                np.column_stack([np.ones(m.sum()), (pg - D)[m]]), Y[m, COL[21]], rcond=None)[0]
        resid = (forward(th, p) - Y) * ob
        records.append({"iteration": iteration, "rows": len(fit),
                        "residual_rmse_by_feature": np.sqrt(np.nanmean(resid * resid, axis=0)).tolist()})
    output.write_bytes(pickle.dumps(p))
    (OUT / "fit_ids.json").write_text(json.dumps(fit.frame if False else fit.row_id.tolist()))
    (OUT / "report.json").write_text(json.dumps({
        "status": "Unlabeled test-distribution parameter adaptation",
        "fit_rows": len(fit), "iterations": records,
        "source_params": "work/params_final.pkl",
        "output": output.relative_to(ROOT).as_posix(),
        "limitation": "Sensor-law fit only; no NDEM score or target label was used."
    }, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
