import numpy as np

from power_intraday_alpha.execution import full_schedule, model_optimal, twap
from power_intraday_alpha.impact import SqrtConditional
from power_intraday_alpha.sim import Simulator


def test_kernel_is_causal():
    sim = Simulator(n_days=2)
    assert np.allclose(np.triu(sim.kernel), 0.0)


def test_cost_increases_with_size():
    sim = Simulator(n_days=2)
    pdy = sim.product_day(0, 12)
    costs = [sim.execute(pdy, full_schedule(96, 12, twap(84, Q)), +1, noise=False) for Q in (10, 50, 200)]
    assert costs[0] < costs[1] < costs[2]


def test_buy_and_sell_symmetric_without_noise():
    sim = Simulator(n_days=2)
    pdy = sim.product_day(1, 3)
    q = full_schedule(96, 12, twap(84, 50))
    assert np.isclose(sim.execute(pdy, q, +1, noise=False), sim.execute(pdy, q, -1, noise=False))


def test_optimal_schedule_is_feasible():
    m = SqrtConditional()
    m.params = np.array([2.5, 0.3, 0.7, 0.9])
    n = 84
    v = np.linspace(5, 40, n)
    r2 = np.linspace(0.3, 1.0, n)
    for lam in (0.0, 1e-3):
        q = model_optimal(m, v, r2, 1 / np.sqrt(v), 100.0, lam)
        assert np.all(q >= 0) and np.isclose(q.sum(), 100.0)
