import numpy as np

from demand_agents.models.uncertainty import ConformalCalibrator, coverage


def test_conformal_intervals_hit_nominal_coverage():
    rng = np.random.default_rng(0)
    n = 4000
    yhat = rng.uniform(2, 30, size=n)
    y = rng.poisson(yhat).astype(float)
    groups = np.array(["a", "b"] * (n // 2))
    cal = ConformalCalibrator(0.9).fit(y[:2000], yhat[:2000], groups[:2000])
    lo, hi = cal.intervals(yhat[2000:], groups[2000:])
    assert 0.86 <= coverage(y[2000:], lo, hi) <= 0.94
    assert (lo >= 0).all() and (hi >= yhat[2000:]).all()


def test_scale_widens_intervals_and_sigma():
    rng = np.random.default_rng(1)
    yhat = rng.uniform(5, 20, 500)
    y = rng.poisson(yhat).astype(float)
    cal = ConformalCalibrator(0.9).fit(y, yhat, np.zeros(500))
    lo1, hi1 = cal.intervals(yhat, np.zeros(500), 1.0)
    lo2, hi2 = cal.intervals(yhat, np.zeros(500), 1.5)
    assert ((hi2 - lo2) >= (hi1 - lo1)).all()
    assert (cal.sigma(yhat, np.zeros(500), 1.5) > cal.sigma(yhat, np.zeros(500), 1.0)).all()
