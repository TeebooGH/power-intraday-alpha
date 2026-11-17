"""The two places where future information could leak into a decision."""
import dataclasses

import numpy as np

from power_intraday_alpha.execution import forecast_curves
from power_intraday_alpha.impact import Profiles
from power_intraday_alpha.sim import Simulator


def test_profiles_only_read_training_days():
    sim = Simulator(n_days=10)
    read = sim.product_day

    def guarded(day, hour):
        assert day < 5, f"profile estimation read test day {day}"
        return read(day, hour)

    sim.product_day = guarded
    Profiles(sim, range(5))


def test_execution_forecast_ignores_data_after_decision():
    sim = Simulator(n_days=10)
    prof = Profiles(sim, range(5))
    pdy = sim.product_day(7, 12)
    start = 12
    # scramble everything the trader cannot have seen at the start of bin `start`
    rng = np.random.default_rng(0)
    v_obs = pdy.v_obs.copy()
    v_obs[start:] = rng.uniform(1, 500, len(v_obs) - start)
    mid = pdy.mid.copy()
    mid[start + 1:] += rng.normal(0, 50, len(mid) - start - 1)
    future_scrambled = dataclasses.replace(pdy, v_obs=v_obs, mid=mid)
    for a, b in zip(forecast_curves(prof, pdy, start), forecast_curves(prof, future_scrambled, start), strict=True):
        assert np.array_equal(a, b)
