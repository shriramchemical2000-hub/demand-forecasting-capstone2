import numpy as np

from demand_agents.inventory.policies import base_stock_components
from demand_agents.inventory.simulator import simulate_base_stock, summarize


def test_window_sums_cover_protection_interval():
    T, N, P = 10, 2, 3
    fc = np.arange(1, T + 1, dtype=float)[:, None] * np.ones((1, N))
    mean_d, safety = base_stock_components(fc, np.zeros((T, N)), 1.645, P)
    assert mean_d.shape == (T + 1, N)
    assert mean_d[0, 0] == 1 + 2 + 3          # days 0..P-1
    assert mean_d[1, 0] == 2 + 3 + 4          # days t+1..t+P for t = 0
    assert (safety == 0).all()


def test_risk_multiplier_scales_safety_stock():
    T, N = 8, 1
    fc, sg = np.full((T, N), 10.0), np.full((T, N), 3.0)
    _, base = base_stock_components(fc, sg, 1.645, 3)
    _, risky = base_stock_components(fc, sg, 1.645, 3, risk=np.full((T, N), 1.5))
    assert np.allclose(risky, 1.5 * base)


def test_simulation_conserves_units_and_never_negative():
    rng = np.random.default_rng(0)
    T, N = 40, 5
    demand = rng.poisson(8, size=(T, N)).astype(float)
    S = np.full((T + 1, N), 40.0)
    sim = simulate_base_stock(demand, S, lead_time_days=2)
    assert np.allclose(sim.sold + sim.lost, demand)
    assert (sim.on_hand_end >= 0).all() and (sim.orders >= 0).all()


def test_more_safety_stock_means_fewer_stockouts():
    rng = np.random.default_rng(1)
    T, N = 120, 6
    lam = np.full((T, N), 10.0)
    demand = rng.poisson(lam).astype(float)
    results = {}
    for z in (0.0, 2.0):
        m, s = base_stock_components(lam, np.sqrt(lam), z, 3)
        sim = simulate_base_stock(demand, m + s, 2)
        results[z] = summarize(sim, demand, np.full(N, 0.01), np.full(N, 1.0))
    assert results[2.0]["lost_units"] < results[0.0]["lost_units"]
    assert results[2.0]["avg_on_hand_units"] > results[0.0]["avg_on_hand_units"]


def test_nowcast_only_reacts_to_significant_deviation():
    from demand_agents.inventory.policies import nowcast_factor

    rng = np.random.default_rng(0)
    actual = rng.poisson(10, (40, 2)).astype(float)
    actual[25:, 0] *= 2                                   # series 0 surges, series 1 stays normal
    f = nowcast_factor(actual, np.full((40, 2), 10.0), np.full((40, 2), 3.2), window=7)
    assert f.shape == (40 - 6 + 1, 2)
    assert f[-1, 0] > 1.5                                 # the real surge is picked up
    assert (np.abs(f[:, 1] - 1.0) < 1e-9).mean() > 0.85   # pure noise is (almost always) ignored
    assert (f >= 0.8).all() and (f <= 2.0).all()


def test_level_adjust_scales_expected_demand_only():
    fc, sg = np.full((6, 1), 10.0), np.full((6, 1), 2.0)
    m0, s0 = base_stock_components(fc, sg, 1.645, 3)
    m1, s1 = base_stock_components(fc, sg, 1.645, 3, level_adjust=np.full((7, 1), 1.5))
    assert np.allclose(m1, 1.5 * m0) and np.allclose(s1, s0)


def test_nowcast_ignores_known_event_periods():
    from demand_agents.inventory.policies import nowcast_factor

    actual = np.full((20, 1), 100.0)
    actual[12, 0] = 0.0                                    # a store-closure day inside the window
    forecast, sigma = np.full((20, 1), 100.0), np.full((20, 1), 5.0)
    plain = nowcast_factor(actual, forecast, sigma, window=7, z_threshold=3.0)
    ignore = np.zeros((20, 1), bool)
    ignore[12, 0] = True
    guarded = nowcast_factor(actual, forecast, sigma, window=7, z_threshold=3.0, ignore=ignore)
    assert plain.min() < 0.95                              # without the guard the closure looks like a demand collapse
    assert np.allclose(guarded, 1.0)                       # with the guard nothing happens
