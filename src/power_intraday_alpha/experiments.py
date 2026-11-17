"""Reproduce every number of the README: `uv run power-intraday-alpha`."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .execution import (
    forecast_curves,
    full_schedule,
    last_hour,
    model_optimal,
    model_selected_window,
    twap,
    volume_weighted,
    window_vw,
)
from .impact import LOOKBACK, MODELS, Profiles, SqrtConditional, simulate_metaorders
from .sim import Simulator

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "results"  # <repo>/results with the src/ layout
N_DAYS, FIRST_TEST, STEP, EXEC_SPLIT = 250, 100, 25, 150
N_BOOT_DELTA = 200


def walk_forward(sim, meta):
    preds = []
    for start in range(FIRST_TEST, N_DAYS, STEP):
        train_days = range(0, start)
        prof = Profiles(sim, train_days)
        tr = prof.features(meta[meta.day < start])
        te = prof.features(meta[(meta.day >= start) & (meta.day < start + STEP)])
        for M in MODELS:
            m = M().fit(tr)
            preds.append(te.assign(model=m.name, pred=m.predict(te), fold=start))
    return pd.concat(preds, ignore_index=True)


def bootstrap_diff(a, b, clusters, n=2000, seed=0):
    """Paired difference mean(a-b) with a day-clustered bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    d = pd.Series(a - b).groupby(clusters).mean()
    boots = [d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(n)]
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def true_regressors(sim, feats):
    """Same columns as `Profiles.features`, but with the simulator's TRUE volume and volatility over each
    order's window: the regression without errors in variables (only possible in simulation)."""
    cache, sig, v = {}, [], []
    for d, h, t0, D in feats[["day", "hour", "t0", "D"]].itertuples(index=False):
        if (d, h) not in cache:
            cache[(d, h)] = sim.product_day(d, h)
        pdy = cache[(d, h)]
        sig.append(np.sqrt((pdy.sigma[t0:t0 + D] ** 2).mean()))
        v.append(pdy.v[t0:t0 + D].mean())
    v = np.asarray(v)
    return feats.assign(sigma_hat=sig, v_hat=v, s_hat=1.0 / np.sqrt(v))


def delta_ci_day_bootstrap(feats, n_boot=N_BOOT_DELTA, seed=0):
    """95% CI of the fitted impact exponent, resampling whole days (orders on the same product-day share
    the same price path, so they are not independent: curve_fit's iid standard errors are too small)."""
    rng = np.random.default_rng(seed)
    by_day = dict(tuple(feats.groupby("day")))
    days = np.array(list(by_day))
    deltas = []
    for _ in range(n_boot):
        sample = pd.concat([by_day[d] for d in rng.choice(days, len(days))])
        try:
            deltas.append(SqrtConditional().fit(sample).params[2])
        except RuntimeError:  # curve_fit did not converge on this resample
            continue
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def main(out: Path = DEFAULT_OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    sim = Simulator(n_days=N_DAYS, seed=0)
    meta = simulate_metaorders(sim, per_product=4, seed=1)
    print(f"{len(meta)} metaorders simulated in {time.time() - t:.0f}s")

    # ---------------------------------------------------------------- 1. calibration, walk-forward OOS
    oos = walk_forward(sim, meta)
    rows = []
    for name, g in oos.groupby("model", sort=False):
        rows.append(dict(model=name,
                         mae_vs_expected_cost=float((g.pred - g.oracle).abs().mean()),
                         mae_vs_realised_cost=float((g.pred - g.cost).abs().mean()),
                         r2_realised=float(1 - ((g.cost - g.pred) ** 2).sum() / ((g.cost - g.cost.mean()) ** 2).sum())))
    calib = pd.DataFrame(rows)
    print(calib.to_string(index=False))

    # parameters of the conditional model on the last fold + recovery of the exponent
    prof_all = Profiles(sim, range(0, EXEC_SPLIT))
    feats = prof_all.features(meta[meta.day < EXEC_SPLIT])
    m3 = SqrtConditional().fit(feats)
    # parameter recovery: same specification fitted on the noise-free expected cost
    m3_oracle = SqrtConditional().fit(feats.assign(cost=feats.oracle))
    # where does the gap to the true delta come from? same fits with the true volume / volatility
    feats_true = true_regressors(sim, feats)
    delta_true_regs = {
        "expected_cost": float(SqrtConditional().fit(feats_true.assign(cost=feats_true.oracle)).params[2]),
        "realised_cost": float(SqrtConditional().fit(feats_true).params[2]),
    }
    delta_ci = delta_ci_day_bootstrap(feats)
    print(f"delta: true regressors {delta_true_regs}, estimated regressors expected "
          f"{m3_oracle.params[2]:.3f} / realised {m3.params[2]:.3f}, day-bootstrap CI {delta_ci}")

    # ---------------------------------------------------------------- 2. execution backtest
    Q, start, K = 100.0, LOOKBACK, sim.p.n_bins
    res = []
    for day in range(EXEC_SPLIT, N_DAYS):
        for hour in range(24):
            pdy = sim.product_day(day, hour)
            v_hat, r2_hat, s_hat = forecast_curves(prof_all, pdy, start)
            n = K - start
            scheds = {
                "TWAP (7 h)": twap(n, Q),
                "Volume-weighted (7 h)": volume_weighted(v_hat, Q),
                "Volume-weighted (last 4 h)": window_vw(v_hat, Q, 48),
                "Volume-weighted (last 2 h)": window_vw(v_hat, Q, 24),
                "Last hour only": last_hour(n, Q),
                "Per-bin optimiser (M3, instantaneous approx.)": model_optimal(m3, v_hat, r2_hat, s_hat, Q, 0.0),
                "M3-selected window (kappa=0.25)": model_selected_window(m3, v_hat, r2_hat, Q, 0.25),
                "M3-selected window (kappa=1)": model_selected_window(m3, v_hat, r2_hat, Q, 1.0),
            }
            for name, q in scheds.items():
                qf = full_schedule(K, start, q)
                res.append(dict(day=day, hour=hour, schedule=name,
                                cost=sim.execute(pdy, qf, +1, noise=True, arrival=start),
                                expected=sim.execute(pdy, qf, +1, noise=False, arrival=start)))
    ex = pd.DataFrame(res)
    piv = ex.pivot_table(index=["day", "hour"], columns="schedule", values="cost")
    exp_piv = ex.pivot_table(index=["day", "hour"], columns="schedule", values="expected")
    summary = []
    for s in piv.columns:
        diff, ci = bootstrap_diff(piv[s].to_numpy(), piv["TWAP (7 h)"].to_numpy(), piv.index.get_level_values(0))
        summary.append(dict(schedule=s, mean_cost=float(piv[s].mean()), sd_cost=float(piv[s].std()),
                            expected_cost=float(exp_piv[s].mean()),
                            diff_vs_twap=diff, diff_ci95=ci))
    ex_sum = pd.DataFrame(summary).sort_values("mean_cost")
    print(ex_sum.to_string(index=False))
    print(f"total {time.time() - t:.0f}s")

    # ---------------------------------------------------------------- 3. outputs
    metrics = dict(
        n_metaorders=len(meta), n_days=N_DAYS, walk_forward=dict(first_test_day=FIRST_TEST, refit_every_days=STEP),
        calibration_oos=calib.to_dict("records"),
        m3_params_on_realised_cost=dict(zip(["a", "Y", "delta", "gamma"], m3.params.tolist(), strict=True),
                                        se=m3.se.tolist()),
        m3_params_on_expected_cost=dict(zip(["a", "Y", "delta", "gamma"], m3_oracle.params.tolist(), strict=True),
                                        se=m3_oracle.se.tolist()),
        true_instantaneous_delta=sim.p.impact_delta,
        delta_recovery=dict(
            true_delta=sim.p.impact_delta,
            true_regressors=delta_true_regs,
            estimated_regressors=dict(expected_cost=float(m3_oracle.params[2]), realised_cost=float(m3.params[2]),
                                      realised_cost_ci95_day_bootstrap=delta_ci, n_boot=N_BOOT_DELTA),
        ),
        execution=dict(position_mwh=Q, start_hours_before_close=float(sim.p.tau[start]),
                       test_days=N_DAYS - EXEC_SPLIT, products=24, results=ex_sum.to_dict("records")),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    tau = sim.p.tau
    ax[0].plot(tau, prof_all.v.mean(0), label="market volume / bin (MW)")
    ax0b = ax[0].twinx()
    ax0b.plot(tau, np.sqrt(prof_all.r2.mean(0)), color="C1", label="sd of 5-min price change (EUR/MWh)")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("hours to gate closure")
    ax[0].set_title("Simulated intraday session")
    ax[0].legend(loc="upper left", fontsize=8)
    ax0b.legend(loc="center left", fontsize=8)
    g = oos[oos.model == SqrtConditional.name]
    g = g.assign(bin=pd.qcut(g.pred, 10, labels=False)).groupby("bin")[["pred", "cost", "oracle"]].mean()
    ax[1].plot(g.pred, g.cost, "o", label="realised (mean per decile)")
    ax[1].plot(g.pred, g.oracle, "s", label="true expected cost")
    lim = [g.pred.min(), g.pred.max()]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlabel("predicted cost (EUR/MWh)")
    ax[1].set_title("M3 out-of-sample calibration")
    ax[1].legend(fontsize=8)
    for _, r in ex_sum.iterrows():
        ax[2].plot(r.sd_cost, r.expected_cost, "o")
        ax[2].annotate(r.schedule.replace(" (M3, instantaneous approx.)", ""), (r.sd_cost, r.expected_cost),
                       fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax[2].set_xlabel("sd of realised shortfall (EUR/MWh)")
    ax[2].set_ylabel("expected cost (EUR/MWh)")
    ax[2].set_title(f"Cost / risk of executing {Q:.0f} MWh (OOS days)")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)


if __name__ == "__main__":
    main()
