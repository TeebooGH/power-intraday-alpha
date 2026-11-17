"""Metaorder dataset, ex-ante liquidity/volatility forecasts and market impact models.

Only information available at order start is used as features (no look-ahead):
seasonal profiles are estimated on training days only, and the day's liquidity/volatility
level is estimated from the trailing `lookback` bins of observed market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .sim import Simulator

DURATIONS = (6, 12, 24, 36)   # bins: 30 min to 3 h
LOOKBACK = 12                  # 1 hour of trailing market data


def simulate_metaorders(sim: Simulator, per_product: int = 4, seed: int = 1) -> pd.DataFrame:
    """TWAP metaorders of random size/duration/start; records realised and oracle cost."""
    rng = np.random.default_rng(seed)
    K = sim.p.n_bins
    rows = []
    for day in range(sim.n_days):
        for hour in range(24):
            pdy = sim.product_day(day, hour)
            r2 = np.diff(pdy.mid) ** 2
            for _ in range(per_product):
                D = int(rng.choice(DURATIONS))
                t0 = int(rng.integers(LOOKBACK, K - D + 1))
                rho = rng.uniform(0.02, 0.30)                       # target participation
                Q = rho * pdy.v_obs[t0 - LOOKBACK:t0].mean() * D    # sized on what the trader sees
                side = int(rng.choice([-1, 1]))
                q = np.zeros(K)
                q[t0:t0 + D] = Q / D
                rows.append(dict(
                    day=day, hour=hour, t0=t0, D=D, Q=Q, side=side,
                    cost=sim.execute(pdy, q, side, noise=True),
                    oracle=sim.execute(pdy, q, side, noise=False),
                    trail_v=pdy.v_obs[t0 - LOOKBACK:t0].mean(),
                    trail_r2=r2[t0 - LOOKBACK:t0].mean(),
                ))
    return pd.DataFrame(rows)


class Profiles:
    """Seasonal (hour x time-to-delivery bin) profiles of volume and squared returns."""

    def __init__(self, sim: Simulator, train_days):
        K = sim.p.n_bins
        V = np.zeros((24, K))
        R2 = np.zeros((24, K))
        for day in train_days:
            for hour in range(24):
                pdy = sim.product_day(day, hour)
                V[hour] += pdy.v_obs
                R2[hour] += np.diff(pdy.mid) ** 2
        n = len(train_days)
        # light smoothing across bins: squared returns are very noisy (jumps)
        kern = np.ones(5) / 5
        self.v = V / n
        self.r2 = np.array([np.convolve(np.pad(r, 2, mode="edge"), kern, "valid") for r in R2 / n])

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ex-ante forecasts over each metaorder's window."""
        out = df.copy()
        sig_hat, v_hat, liq_hat, vol_hat = [], [], [], []
        for h, t0, D, tv, tr in df[["hour", "t0", "D", "trail_v", "trail_r2"]].itertuples(index=False):
            lv = tv / self.v[h, t0 - LOOKBACK:t0].mean()
            lr = np.clip(tr / self.r2[h, t0 - LOOKBACK:t0].mean(), 0.25, 4.0)
            v_hat.append(lv * self.v[h, t0:t0 + D].mean())
            sig_hat.append(np.sqrt(lr * self.r2[h, t0:t0 + D].mean()))
            liq_hat.append(lv)
            vol_hat.append(np.sqrt(lr))
        out["v_hat"], out["sigma_hat"] = v_hat, sig_hat
        out["liq_hat"], out["vol_hat"] = liq_hat, vol_hat
        out["s_hat"] = 1.0 / np.sqrt(out["v_hat"])   # spread proxy ~ 1/sqrt(liquidity)
        return out


# ----------------------------------------------------------------------------- models
class ConstantCost:
    name = "M0 constant cost"

    def fit(self, X):
        self.c = X["cost"].mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.c)


class LinearImpact:
    """Almgren-Chriss style: cost linear in size, constant liquidity."""
    name = "M1 linear in Q"

    def fit(self, X):
        A = np.c_[np.ones(len(X)), X["Q"]]
        self.b = np.linalg.lstsq(A, X["cost"], rcond=None)[0]
        return self

    def predict(self, X):
        return self.b[0] + self.b[1] * X["Q"].to_numpy()


class SqrtUnconditional:
    """Square-root law with market-wide constant sigma and volume (ignores time-to-delivery)."""
    name = "M2 sqrt law, constant liquidity"

    def fit(self, X):
        A = np.c_[np.ones(len(X)), np.sqrt(X["Q"])]
        self.b = np.linalg.lstsq(A, X["cost"], rcond=None)[0]
        return self

    def predict(self, X):
        return self.b[0] + self.b[1] * np.sqrt(X["Q"].to_numpy())


class SqrtConditional:
    """cost = a * s_hat + Y * sigma_hat * (rate / v_hat)^delta * D^gamma

    rate = Q / D is the trading rate (MW per bin), v_hat / sigma_hat are ex-ante forecasts of
    market volume and volatility over the order's window, conditioned on time to delivery,
    product hour and the day's observed activity. D^gamma captures the build-up of transient
    impact over the order's duration (gamma = 0 would mean purely instantaneous impact)."""
    name = "M3 participation-rate law, liquidity-conditioned"

    @staticmethod
    def _f(Z, a, y, d, g):
        s, sig, x, D = Z
        return a * s + y * sig * x ** d * D ** g

    @staticmethod
    def _z(X):
        return (X["s_hat"].to_numpy(), X["sigma_hat"].to_numpy(),
                (X["Q"] / X["D"] / X["v_hat"]).to_numpy(), X["D"].to_numpy(float))

    def fit(self, X):
        self.params, cov = curve_fit(self._f, self._z(X), X["cost"].to_numpy(),
                                     p0=(1.0, 1.0, 0.5, 0.5), maxfev=20000)
        self.se = np.sqrt(np.diag(cov))
        return self

    def predict(self, X):
        return self._f(self._z(X), *self.params)


MODELS = (ConstantCost, LinearImpact, SqrtUnconditional, SqrtConditional)
