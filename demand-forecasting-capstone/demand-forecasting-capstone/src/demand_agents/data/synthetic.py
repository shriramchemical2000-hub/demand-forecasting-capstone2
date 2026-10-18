"""Synthetic M5-like retail data with *known* demand shocks.

Why synthetic first? It lets the team develop and test the whole agent pipeline
before the real datasets are wired in, and the ground-truth ``shock_type`` column
lets us measure how well the Critic detects shocks.

Shocks injected inside the test window (the last ``test_days`` days):
  * heatwave : temperature exceeds anything seen in training -> temperature-sensitive
               items surge. Tree models cannot extrapolate, so forecasts lag behind.
  * event    : a major event with an uplift whose size is not in any feature.
  * surge    : an unobserved +60% demand surge on a subset of series (no feature
               explains it; only recent forecast errors reveal it).
Regular holidays (event flag) and price promotions also appear throughout history.
"""
import numpy as np
import pandas as pd

HOLIDAYS_MD = [(1, 1), (2, 14), (5, 27), (7, 4), (9, 1), (11, 25), (12, 25)]
WEEKDAY_WEIGHTS = np.array([0.85, 0.85, 0.90, 0.95, 1.10, 1.35, 1.20])


def generate_synthetic(
    n_stores: int = 3,
    n_items: int = 12,
    n_days: int = 1000,
    start_date: str = "2021-01-01",
    test_days: int = 56,
    seed: int = 42,
    shocks: bool = True,
) -> pd.DataFrame:
    """``shocks=False`` creates a *calm* world: same demand process but no heatwave, surge or major event in the test window."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    T = n_days
    test_start = T - test_days
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values

    stores = [f"S{i + 1}" for i in range(n_stores)]
    items = [f"I{i + 1:02d}" for i in range(n_items)]
    n_sensitive = max(1, int(round(0.4 * n_items)))
    category = {it: ("beverages" if i < n_sensitive else "staples") for i, it in enumerate(items)}

    # ---- weather (common AR(1) noise + store offset) -------------------------------
    noise = np.zeros(T)
    for t in range(1, T):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 2.0)
    season = 18 + 7 * np.sin(2 * np.pi * (doy - 110) / 365)
    heat_days = np.zeros(T, dtype=bool)
    if shocks:
        heat_days[test_start + 14 : test_start + 24] = True
    store_offset = rng.normal(0, 2.0, size=n_stores)
    temp_store = season[:, None] + noise[:, None] + store_offset[None, :]  # (T, Ns)
    temp_store[heat_days] += 10.0

    # ---- events (holidays + one major event in the test window) -------------------
    is_event = np.array([(d.month, d.day) in HOLIDAYS_MD for d in dates])
    major = np.zeros(T, dtype=bool)
    if shocks:
        major[test_start + 35 : test_start + 38] = True
    is_event = is_event | major
    event_uplift = np.ones(T)
    t = 0
    while t < T:
        if is_event[t]:
            end = t
            while end + 1 < T and is_event[end + 1]:
                end += 1
            event_uplift[t : end + 1] = rng.uniform(1.3, 2.2)
            t = end + 1
        else:
            t += 1
    item_event_resp = rng.uniform(0.7, 1.3, size=n_items)

    # ---- per series arrays ----------------------------------------------------------
    n_series = n_stores * n_items
    surge_series = set(rng.choice(n_series, size=max(1, n_series // 4), replace=False).tolist())
    surge_days = np.zeros(T, dtype=bool)
    if shocks:
        surge_days[test_start + 30 : test_start + 44] = True

    base_price = rng.uniform(2, 15, size=n_items)
    records = []
    k = 0
    for si, store in enumerate(stores):
        for ii, item in enumerate(items):
            level = float(np.clip(np.exp(rng.normal(np.log(8), 0.6)), 1.5, 40))
            price_factor = 1 + rng.uniform(-0.03, 0.03)
            depth = np.zeros(T)
            for s in np.where(rng.random(T) < 0.02)[0]:
                depth[s : s + 7] = np.maximum(depth[s : s + 7], rng.uniform(0.15, 0.30))
            price = base_price[ii] * price_factor * (1 - depth)

            weekly = WEEKDAY_WEIGHTS[dow] / WEEKDAY_WEIGHTS.mean()
            yearly = 1 + 0.12 * np.sin(2 * np.pi * (doy - 30) / 365)
            trend = 1 + 0.0002 * np.arange(T)
            promo = (1 - depth) ** (-1.6)
            ev_mult = 1 + (event_uplift - 1) * item_event_resp[ii] * is_event
            sensitive = category[item] == "beverages"
            temp_mult = np.exp(0.04 * (temp_store[:, si] - 18)) if sensitive else np.ones(T)
            in_surge = k in surge_series
            surge = np.where(surge_days & in_surge, 1.6, 1.0)

            lam = level * weekly * yearly * trend * promo * ev_mult * temp_mult * surge
            sales = rng.poisson(lam * rng.gamma(25, 1 / 25, size=T))

            shock = np.full(T, "", dtype=object)
            shock[is_event & (np.arange(T) >= test_start)] = "event"
            if sensitive:
                shock[heat_days] = "heatwave"
            if in_surge:
                shock[surge_days] = "surge"

            records.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store_id": store,
                        "item_id": item,
                        "series_id": f"{store}_{item}",
                        "category": category[item],
                        "sales": sales,
                        "sell_price": price.round(2),
                        "is_event": is_event.astype(int),
                        "temp": temp_store[:, si].round(1),
                        "shock_type": shock,
                    }
                )
            )
            k += 1

    df = pd.concat(records, ignore_index=True)
    df["shock_type"] = df["shock_type"].astype(str)
    df["is_shock"] = df["shock_type"] != ""
    return df


WALMART_HOLIDAY_WEEKS = [
    "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
    "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
    "2012-02-10", "2012-09-07", "2012-11-23", "2012-12-28",
]


def generate_walmart_like(n_stores: int = 45, n_weeks: int = 143, shock_weeks: int = 8, seed: int = 42) -> pd.DataFrame:
    """Weekly store-level sales in the *exact column format* of the Kaggle Walmart file:
    Store, Date (dd-mm-YYYY), Weekly_Sales, Holiday_Flag, Temperature (F), Fuel_Price, CPI, Unemployment.

    The last ``shock_weeks`` weeks contain an unobserved +18% surge in about a third of the stores.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="7D")
    holiday = dates.isin(pd.to_datetime(WALMART_HOLIDAY_WEEKS)).astype(int)
    woy = dates.isocalendar().week.to_numpy().astype(int)
    surge_stores = set(rng.choice(n_stores, size=max(1, n_stores // 3), replace=False).tolist())
    fuel = 2.6 + 0.6 * np.sin(np.arange(n_weeks) / 30) + rng.normal(0, 0.03, n_weeks).cumsum() * 0.2
    cpi0 = rng.uniform(126, 220, n_stores)
    unemp0 = rng.uniform(5, 14, n_stores)
    rows = []
    for st in range(n_stores):
        level = float(np.clip(np.exp(rng.normal(np.log(1.0e6), 0.45)), 2.5e5, 2.8e6))
        temp_f = 55 + 25 * np.sin(2 * np.pi * (woy - 15) / 52) + rng.normal(0, 4, n_weeks) + rng.normal(0, 6)
        season = 1 + 0.06 * np.sin(2 * np.pi * (woy - 20) / 52) + 0.30 * np.exp(-0.5 * ((woy - 51) / 1.5) ** 2)
        hol = 1 + 0.07 * holiday + 0.20 * np.isin(woy, [47, 48]).astype(float)
        trend = 1 + 0.0004 * np.arange(n_weeks)
        shock = np.ones(n_weeks)
        if st in surge_stores:
            shock[-shock_weeks:] = 1.18
        sales = level * season * hol * trend * shock * np.exp(rng.normal(0, 0.045, n_weeks))
        for t in range(n_weeks):
            rows.append(
                {
                    "Store": st + 1,
                    "Date": dates[t].strftime("%d-%m-%Y"),
                    "Weekly_Sales": round(float(sales[t]), 2),
                    "Holiday_Flag": int(holiday[t]),
                    "Temperature": round(float(temp_f[t]), 2),
                    "Fuel_Price": round(float(fuel[t]), 3),
                    "CPI": round(float(cpi0[st] + 0.02 * t), 6),
                    "Unemployment": round(float(unemp0[st] - 0.004 * t), 3),
                }
            )
    return pd.DataFrame(rows)
