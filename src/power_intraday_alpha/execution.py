"""Execution schedules for a fixed position, evaluated in the true simulator."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .impact import LOOKBACK, Profiles, SqrtConditional
from .sim import ProductDay, Simulator


def forecast_curves(prof: Profiles, pdy: ProductDay, start: int):
    """Ex-ante volume / variance / spread curves for the rest of the session, seen at `start`."""
    h = pdy.hour
    lv = pdy.v_obs[start - LOOKBACK:start].mean() / prof.v[h, start - LOOKBACK:start].mean()
    r2 = np.diff(pdy.mid[:start + 1])[-LOOKBACK:] ** 2
    lr = np.clip(r2.mean() / prof.r2[h, start - LOOKBACK:start].mean(), 0.25, 4.0)
    v_hat = lv * prof.v[h, start:]
    r2_hat = lr * prof.r2[h, start:]
    return v_hat, r2_hat, 1.0 / np.sqrt(v_hat)


def twap(n, Q):
    return np.full(n, Q / n)


def volume_weighted(v_hat, Q):
    return Q * v_hat / v_hat.sum()


def last_hour(n, Q, bins=12):
    q = np.zeros(n)
    q[-bins:] = Q / bins
    return q


def model_optimal(model: SqrtConditional, v_hat, r2_hat, s_hat, Q, risk_aversion=0.0):
    """argmin_q  sum_t q_t [a s_t + c sig_t (q_t/v_t)^d] + lambda * sum_t r2_t x_t^2
    x_t = inventory still to trade at the start of bin t; q >= 0, sum q = Q.
    c = Y * n^gamma treats the transient build-up as uniform over the n-bin horizon
    (approximation: the optimiser does not see the decay kernel)."""
    a, y, d, g = model.params
    sig = np.sqrt(r2_hat)
    n = len(v_hat)
    c = y * n**g * sig * v_hat ** (-d)
    lin = a * s_hat

    def obj(z):
        q = Q * z
        x = Q - np.concatenate([[0.0], np.cumsum(q)[:-1]])
        J = (q * lin + c * q ** (1 + d)).sum() + risk_aversion * (r2_hat * x**2).sum()
        # gradient: dx_t/dq_u = -1 for t > u
        w = 2 * risk_aversion * r2_hat * x
        tail = np.concatenate([np.cumsum(w[::-1])[::-1][1:], [0.0]])
        grad = lin + (1 + d) * c * q**d - tail
        return J, Q * grad

    # risk-neutral case: KKT gives q_t = ((mu - lin_t) / ((1+d) c_t))^(1/d), mu found by bisection
    def q_of(mu):
        return np.clip((mu - lin) / ((1 + d) * c), 0, None) ** (1 / d)
    lo, hi = lin.min(), lin.max() + (1 + d) * c.max() * Q**d
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        lo, hi = (mu, hi) if q_of(mu).sum() < Q else (lo, mu)
    q0 = q_of(hi)
    if risk_aversion == 0.0:
        return Q * q0 / q0.sum()

    z0 = q0 / q0.sum()
    res = minimize(obj, z0, jac=True, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1, "jac": lambda z: np.ones(n)}],
                   options={"maxiter": 300, "ftol": 1e-12})
    z = np.clip(res.x, 0, None)
    return Q * z / z.sum()


def model_selected_window(model: SqrtConditional, v_hat, r2_hat, Q, kappa=0.25,
                          windows=(12, 24, 48, 84)):
    """Pick the execution window (last W bins, volume-weighted inside) minimising
    M3-predicted mean cost + kappa * predicted sd of the shortfall (both EUR/MWh).
    Uses the metaorder-level model directly, so the duration term D^gamma is accounted for."""
    n = len(v_hat)
    best, best_q = np.inf, None
    for W in windows:
        W = min(W, n)
        q = np.zeros(n)
        q[-W:] = Q * v_hat[-W:] / v_hat[-W:].sum()
        vw = v_hat[-W:].mean()
        feats = (np.array([1 / np.sqrt(vw)]), np.array([np.sqrt(r2_hat[-W:].mean())]),
                 np.array([Q / W / vw]), np.array([float(W)]))
        mean = model._f(feats, *model.params)[0]
        x = Q - np.concatenate([[0.0], np.cumsum(q)[:-1]])
        sd = np.sqrt((r2_hat * x**2).sum()) / Q
        if mean + kappa * sd < best:
            best, best_q = mean + kappa * sd, q
    return best_q


def full_schedule(K, start, q_tail):
    q = np.zeros(K)
    q[start:] = q_tail
    return q


def window_vw(v_hat, Q, W):
    q = np.zeros(len(v_hat))
    q[-W:] = Q * v_hat[-W:] / v_hat[-W:].sum()
    return q
