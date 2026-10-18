"""Leakage-safe feature engineering for daily or weekly data.

Forecasts are made ``horizon`` periods ahead, so every feature that depends on past
sales must be lagged by at least ``horizon`` periods. Calendar, price, weather and
other exogenous features are treated as known in advance (weather = forecast ~ actual).
"""
import pandas as pd

CATEGORICAL = ["store_code", "item_code"]


def build_features(
    df: pd.DataFrame,
    horizon: int,
    lags: list[int],
    windows: list[int],
    freq: str = "D",
    extra_features=(),
    price_window: int = 56,
):
    """Return (frame_with_features, feature_names)."""
    bad = [k for k in lags if k < horizon]
    if bad:
        raise ValueError(f"lags {bad} are shorter than the forecast horizon ({horizon}) -> leakage")

    df = df.sort_values(["series_id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("series_id", sort=False)
    feats: list[str] = []

    for k in lags:
        df[f"lag_{k}"] = g["sales"].shift(k)
        feats.append(f"lag_{k}")

    shifted = g["sales"].shift(horizon)
    by_series = shifted.groupby(df["series_id"], sort=False)
    for w in windows:
        mp = max(2, w // 2)
        df[f"roll_mean_{w}"] = by_series.transform(lambda s: s.rolling(w, min_periods=mp).mean())
        df[f"roll_std_{w}"] = by_series.transform(lambda s: s.rolling(w, min_periods=mp).std())
        feats += [f"roll_mean_{w}", f"roll_std_{w}"]

    d = pd.to_datetime(df["date"])
    df["month"] = d.dt.month
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    feats += ["month", "week_of_year", "is_event"]
    if freq == "D":
        df["dow"] = d.dt.dayofweek
        df["day_of_month"] = d.dt.day
        feats += ["dow", "day_of_month"]

    if df["sell_price"].nunique(dropna=True) > 1:  # skip price features when price is constant / absent
        price_ref = g["sell_price"].transform(lambda s: s.shift(1).rolling(price_window, min_periods=1).mean())
        df["sell_price_feat"] = df["sell_price"]
        df["price_rel"] = df["sell_price"] / price_ref
        feats += ["sell_price_feat", "price_rel"]

    df["temp_feat"] = df["temp"]
    df["temp_roll"] = g["temp"].transform(lambda s: s.rolling(7 if freq == "D" else 3, min_periods=1).mean())
    feats += ["temp_feat", "temp_roll"]

    for col in extra_features:
        if col in df.columns:
            feats.append(col)

    df["store_code"] = df["store_id"].astype("category").cat.codes
    df["item_code"] = df["item_id"].astype("category").cat.codes
    feats += CATEGORICAL
    return df, feats
