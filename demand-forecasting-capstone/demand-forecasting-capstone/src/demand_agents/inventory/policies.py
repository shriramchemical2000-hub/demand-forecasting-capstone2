"""Base-stock (order-up-to) policy.

Timeline convention (days are 0..T-1 inside the evaluation window):
  * An order placed at the END of day t arrives at the START of day t + 1 + L
    (L = lead_time_days; L = 0 means next-morning delivery).
  * Reviews happen every R days (R = 1 here), so the inventory position must cover
    demand over the protection interval P = L + R days: days t+1 ... t+P.
  * Order-up-to level  S_t = sum(forecast over P days) + z * sqrt(sum(sigma_d^2 * risk_d^2)).

Forecasts for days t+1..t+P are available at the end of day t as long as P <= horizon
(the forecast model is built for a 7-day-ahead horizon).
"""
import numpy as np


def protection_days(lead_time_days: int, review_period_days: int = 1) -> int:
    return int(lead_time_days + review_period_days)


def base_stock_components(forecast, sigma, z: float, P: int, risk=None, level_adjust=None):
    """forecast, sigma, risk: arrays (T, N) for days 0..T-1.

    Returns (mean_demand, safety_stock), both (T+1, N). Row 0 is the level at the end of
    day -1 (initial stock); row t+1 is the level for the end-of-day-t decision.
    """
    forecast = np.asarray(forecast, float)
    sigma = np.asarray(sigma, float)
    risk = np.ones_like(forecast) if risk is None else np.asarray(risk, float)
    T, N = forecast.shape

    def pad(a):  # repeat the last row so late-window decisions still have P days ahead
        return np.vstack([a, np.repeat(a[-1:], P, axis=0)])

    def window_sums(a):
        c = np.vstack([np.zeros((1, N)), np.cumsum(a, axis=0)])
        return c[P : P + T + 1] - c[: T + 1]

    mean_demand = window_sums(pad(forecast))
    if level_adjust is not None:
        mean_demand = mean_demand * np.asarray(level_adjust, float)
    variance = window_sums(pad((sigma * risk) ** 2))
    return mean_demand, z * np.sqrt(variance)


def nowcast_factor(actual, forecast, sigma, window: int, z_threshold: float = 2.0, shrink: float = 0.7,
                   clip=(0.8, 2.0), ignore=None):
    """Re-centre expected demand when *recently observed* demand is significantly off the forecast.

    actual, forecast, sigma: arrays (K + T, N) where the first K - 1 + ... rows are history (the rows just
    before the evaluation window) and the last T rows are the window itself. At the end of period t the
    realised demand of the last ``window`` periods is known, so
        z = (sum actual - sum forecast) / sqrt(sum sigma^2)          over the last ``window`` periods
    and, only if |z| > z_threshold, the factor is 1 + shrink * (sum actual / sum forecast - 1), clipped.
    Returns an array (T + 1, N); row 0 (before the window starts) is 1.
    """
    actual, forecast, sigma = (np.asarray(a, float) for a in (actual, forecast, sigma))
    if ignore is not None:  # known events (holidays, store closures) are already in the forecast: they must not trigger a re-centring
        ig = np.asarray(ignore, bool)
        actual, forecast, sigma = np.where(ig, 0.0, actual), np.where(ig, 0.0, forecast), np.where(ig, 0.0, sigma)
    total, N = actual.shape
    T = total - (window - 1)
    if T < 1:
        raise ValueError("need at least window - 1 rows of history before the window")

    def roll(a):
        c = np.vstack([np.zeros((1, N)), np.cumsum(a, axis=0)])
        return c[window:] - c[:-window]

    sa, sf, sv = roll(actual), roll(forecast), roll(sigma**2)
    z = (sa - sf) / np.sqrt(np.maximum(sv, 1e-12))
    ratio = np.where(sf > 0, sa / np.maximum(sf, 1e-12), 1.0)
    factor = np.where(np.abs(z) > z_threshold, 1 + shrink * (ratio - 1), 1.0)
    factor = np.clip(factor, clip[0], clip[1])
    return np.vstack([np.ones((1, N)), factor[-T:]])
