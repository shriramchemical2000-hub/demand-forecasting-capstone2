"""Classical baselines the agentic system must beat.

All baselines are evaluated on the same test dates. ETS/ARIMA are refit at the start of every
forecast block (rolling origin) and forecast 1..horizon periods ahead. This is slightly *easier*
than the GBM (always ``horizon`` periods ahead), so comparisons are conservative.
"""
import warnings

import numpy as np
import pandas as pd


def seasonal_naive(panel: pd.DataFrame, horizon: int, period: int = 7) -> pd.Series:
    """Demand one (or more) seasonal cycles earlier, at least ``horizon`` periods old."""
    k = period * int(np.ceil(horizon / period))
    return panel.sort_values(["series_id", "date"]).groupby("series_id")["sales"].shift(k)


def _ets_fit_factory(period: int):
    def _fit(y: np.ndarray, h: int):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        seasonal = "add" if len(y) >= 2 * period + 2 else None  # need two full cycles to estimate seasonality
        fit = ExponentialSmoothing(
            y, trend=None, seasonal=seasonal, seasonal_periods=period if seasonal else None,
            initialization_method="estimated",
        ).fit()
        return fit.forecast(h), float(np.std(fit.resid))

    return _fit


def _arima_fit_factory(order):
    def _fit(y: np.ndarray, h: int):
        from statsmodels.tsa.arima.model import ARIMA

        fit = ARIMA(y, order=tuple(order)).fit()
        return fit.forecast(h), float(np.std(fit.resid))

    return _fit


def rolling_classical_forecast(
    panel: pd.DataFrame, test_start, horizon: int, method: str = "ets", arima_order=(7, 0, 0),
    window: int = 365, season_period: int = 7, sigma_at=None,
):
    """Rolling-origin forecasts for the test window.

    Returns (DataFrame[series_id, date, y_hat], {series_id: one-step residual std}). The residual std comes
    from the first fit whose forecasts reach ``sigma_at`` (default: ``test_start``), i.e. the same kind of fit
    the classical policy relies on during the test window.
    """
    fit_fn = _ets_fit_factory(season_period) if method == "ets" else _arima_fit_factory(arima_order)
    wide = panel.pivot(index="date", columns="series_id", values="sales").sort_index()
    dates = wide.index
    test_dates = dates[dates >= pd.Timestamp(test_start)]
    first = dates.get_loc(test_dates[0])
    sigma_idx = dates.get_loc(dates[dates >= pd.Timestamp(sigma_at if sigma_at is not None else test_start)][0])

    rows, sigma = [], {}
    n_fallback = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sid in wide.columns:
            y = wide[sid].to_numpy(dtype=float)
            out = np.full(len(test_dates), np.nan)
            for o in range(0, len(test_dates), horizon):
                hist = y[max(0, first + o - window) : first + o]
                hist = hist[np.isfinite(hist)]  # items launched late have no history before launch
                fc, sd = None, None
                if len(hist) >= 2 * season_period:
                    try:
                        fc, sd = fit_fn(hist, horizon)
                        fc = np.asarray(fc, float)
                    except Exception:
                        fc = None
                # statsmodels can return NaN/inf without raising: treat that as a failed fit
                if fc is None or not np.all(np.isfinite(fc)) or sd is None or not np.isfinite(sd):
                    base = hist[-season_period:] if len(hist) else np.zeros(1)
                    fc = np.resize(base, horizon).astype(float)
                    sd = float(np.std(hist)) if len(hist) > 1 else 1.0
                    n_fallback += 1
                if sid not in sigma and first + o + horizon > sigma_idx:
                    sigma[sid] = max(sd, 1e-6) if np.isfinite(sd) else 1.0
                take = min(horizon, len(test_dates) - o)
                out[o : o + take] = np.clip(fc[:take], 0, None)
            rows.append(pd.DataFrame({"series_id": sid, "date": test_dates, "y_hat": out}))
    if n_fallback:
        warnings.warn(f"{method}: {n_fallback} fits were unusable and replaced by a seasonal-naive forecast")
    return pd.concat(rows, ignore_index=True), sigma
