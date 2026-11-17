"""Synthetic continuous intraday power market (one hourly product = one limit-order-book session).

Everything here is an ASSUMPTION of the simulator, chosen to reproduce public stylized facts
of European continuous intraday markets (volume concentrated in the last hours before gate
closure, volatility rising towards delivery, hourly renewable-forecast updates causing jumps,
market volumes growing year on year). No proprietary or real order-book data is used.

Time grid: K bins of `bin_minutes` covering the last `session_hours` before gate closure.
tau[k] = time to gate closure (hours) at the start of bin k.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MarketParams:
    session_hours: float = 8.0
    bin_minutes: int = 5
    # volume per bin (MW traded by the market per 5-min bin) near gate closure
    v0: float = 40.0
    v_floor: float = 0.15          # baseline liquidity far from delivery (fraction of v0)
    tau_v: float = 1.5             # liquidity build-up time scale (hours)
    volume_trend: float = 0.40     # +40% market volume from first to last simulated day
    peak_liquidity: float = 1.3    # peak hours (8h-20h) are more liquid
    # volatility per bin (EUR/MWh)
    sigma0: float = 0.6
    sigma_boost: float = 0.6       # sigma(0) = sigma0 * (1 + boost)
    tau_sigma: float = 1.0
    # renewable forecast updates: one potential jump per hour
    jump_prob: float = 0.5
    jump_scale: float = 2.0
    # half spread (EUR/MWh) scales like 1/sqrt(liquidity)
    half_spread0: float = 0.3
    # day-level regimes (log-normal), correlated: uncertain days are both busier and more volatile
    regime_sd: float = 0.3
    regime_corr: float = 0.5
    obs_volume_noise: float = 0.25
    # TRUE impact model (unknown to the analyst): propagator with power-law instantaneous impact
    impact_y: float = 0.8          # f = Y * sigma_t * (q_t / v_t)^delta
    impact_delta: float = 0.5
    kernel_k0: float = 2.0         # G(k) = (1 + k/k0)^(-beta), k in bins
    kernel_beta: float = 0.5

    @property
    def n_bins(self) -> int:
        return int(round(self.session_hours * 60 / self.bin_minutes))

    @property
    def tau(self) -> np.ndarray:
        k = np.arange(self.n_bins)
        return (self.n_bins - k) * self.bin_minutes / 60.0


@dataclass
class ProductDay:
    """Market state for one (day, hourly product)."""
    day: int
    hour: int
    v: np.ndarray            # true expected market volume per bin (MW)
    v_obs: np.ndarray        # observed market volume per bin
    sigma: np.ndarray        # true volatility per bin
    half_spread: np.ndarray
    mid: np.ndarray          # fundamental mid path, length K+1 (mid[k] = price at start of bin k)
    liq: float = 1.0
    vol: float = 1.0


@dataclass
class Simulator:
    p: MarketParams = field(default_factory=MarketParams)
    n_days: int = 250
    seed: int = 0

    def __post_init__(self):
        p = self.p
        K = p.n_bins
        k = np.arange(K)
        lag = k[:, None] - k[None, :]
        self.kernel = np.where(lag > 0, (1.0 + np.maximum(lag, 0) / p.kernel_k0) ** (-p.kernel_beta), 0.0)
        self.v_shape = p.v_floor + np.exp(-p.tau / p.tau_v)
        self.sigma_shape = 1.0 + p.sigma_boost * np.exp(-p.tau / p.tau_sigma)
        rng = np.random.default_rng(self.seed)
        c = p.regime_corr
        z1 = rng.standard_normal(self.n_days)
        z2 = c * z1 + np.sqrt(1 - c**2) * rng.standard_normal(self.n_days)
        self.day_liq = np.exp(p.regime_sd * z1 - p.regime_sd**2 / 2)
        self.day_vol = np.exp(p.regime_sd * z2 - p.regime_sd**2 / 2)

    def product_day(self, day: int, hour: int) -> ProductDay:
        p, K = self.p, self.p.n_bins
        rng = np.random.default_rng([self.seed, day, hour])
        trend = 1.0 + p.volume_trend * day / max(self.n_days - 1, 1)
        peak = p.peak_liquidity if 8 <= hour < 20 else 1.0
        liq = self.day_liq[day] * rng.lognormal(-0.01, 0.15)
        vol = self.day_vol[day] * rng.lognormal(-0.01, 0.15)
        v = p.v0 * trend * peak * liq * self.v_shape
        v_obs = v * rng.lognormal(-p.obs_volume_noise**2 / 2, p.obs_volume_noise, K)
        sigma = p.sigma0 * vol * self.sigma_shape
        half_spread = p.half_spread0 * np.sqrt(p.v0 / v)
        steps = sigma * rng.standard_normal(K)
        bins_per_hour = int(60 / p.bin_minutes)
        is_update = (np.arange(K) % bins_per_hour) == 0
        jumps = is_update * (rng.random(K) < p.jump_prob) * rng.normal(0.0, p.jump_scale * vol, K)
        mid = 100.0 + np.concatenate([[0.0], np.cumsum(steps + jumps)])
        return ProductDay(day, hour, v, v_obs, sigma, half_spread, mid, liq, vol)

    def execute(self, pdy: ProductDay, schedule: np.ndarray, side: int, noise: bool = True,
                arrival: int | None = None) -> float:
        """Implementation shortfall (EUR/MWh, positive = cost) of trading `schedule` (MW per bin),
        measured against the mid at `arrival` (decision time; default = first traded bin).

        With noise=False the fundamental path is frozen: the result is the expected cost
        (spread + impact), i.e. the oracle the impact model tries to predict.
        """
        p = self.p
        q = np.asarray(schedule, float)
        Q = q.sum()
        active = np.nonzero(q > 0)[0]
        t0 = active[0] if arrival is None else arrival
        f = p.impact_y * pdy.sigma * (q / pdy.v) ** p.impact_delta * side
        impact_before = self.kernel @ f
        mid = pdy.mid[:-1] if noise else np.full(p.n_bins, pdy.mid[t0])
        exec_px = mid + impact_before + side * (pdy.half_spread + 0.5 * np.abs(f))
        vwap = (q * exec_px).sum() / Q
        return side * (vwap - mid[t0])
