"""Stage 4 - exploratory analysis tables. All measures are descriptive and use only the data passed in."""
import numpy as np
import pandas as pd

from ..data.audit import demand_classes


def with_relative_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``rel`` = sales / the series' mean sales, so series of different size can be compared and pooled."""
    d = df.copy()
    m = d.groupby("series_id")["sales"].transform("mean")
    d["rel"] = np.where(m > 0, d["sales"] / m, np.nan)
    return d


def aggregate_series(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["sales"].sum()


def weekday_profile(d: pd.DataFrame) -> pd.DataFrame:
    x = d.assign(dow=pd.to_datetime(d["date"]).dt.dayofweek).groupby("dow")["rel"].mean().reset_index()
    x["weekday"] = x["dow"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    return x


def month_profile(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(month=pd.to_datetime(d["date"]).dt.month).groupby("month")["rel"].mean().reset_index()


def event_effect(d: pd.DataFrame) -> dict:
    if "is_event" not in d.columns or d["is_event"].nunique() < 2:
        return {"available": False}
    ev, non = d.loc[d["is_event"] == 1, "rel"].mean(), d.loc[d["is_event"] == 0, "rel"].mean()
    per = d.groupby(["series_id", "is_event"])["rel"].mean().unstack()
    share_up = float((per[1] > per[0]).mean()) if {0, 1} <= set(per.columns) else float("nan")
    return {"available": True, "event_mean_rel": float(ev), "non_event_mean_rel": float(non), "uplift": float(ev / non - 1),
            "share_series_with_uplift": share_up, "event_day_share": float(d.drop_duplicates("date")["is_event"].mean())}


def price_response(d: pd.DataFrame, min_obs: int = 60) -> dict:
    """Within each series: slope of log(1 + sales) on log(price / mean price). A negative slope means demand falls when price rises."""
    if "sell_price" not in d.columns or d["sell_price"].nunique() < 2:
        return {"available": False}
    slopes = []
    for _, g in d.dropna(subset=["sell_price"]).groupby("series_id"):
        if len(g) < min_obs or g["sell_price"].nunique() < 3:
            continue
        x = np.log(g["sell_price"] / g["sell_price"].mean())
        if x.std() < 1e-6:
            continue
        y = np.log1p(g["sales"])
        slopes.append(float(np.cov(x, y)[0, 1] / x.var()))
    if not slopes:
        return {"available": False}
    s = np.array(slopes)
    return {"available": True, "n_series": len(s), "median_slope": float(np.median(s)), "share_negative": float((s < 0).mean())}


def weather_response(d: pd.DataFrame, min_obs: int = 60) -> dict:
    """Correlation, within each series, between the temperature anomaly (vs its monthly mean) and relative sales."""
    if "temp" not in d.columns or d["temp"].notna().mean() < 0.5:
        return {"available": False}
    x = d.dropna(subset=["temp", "rel"]).copy()
    x["month"] = pd.to_datetime(x["date"]).dt.month
    x["anom"] = x["temp"] - x.groupby(["store_id", "month"])["temp"].transform("mean")
    # compare like with like: sales relative to the series' own monthly level
    x["rel_m"] = x["sales"] / x.groupby(["series_id", "month"])["sales"].transform("mean").replace(0, np.nan)
    corrs = []
    for _, g in x.groupby("series_id"):
        g = g.dropna(subset=["rel_m"])
        if len(g) >= min_obs and g["anom"].std() > 0 and g["rel_m"].std() > 0:
            corrs.append(float(np.corrcoef(g["anom"], g["rel_m"])[0, 1]))
    if not corrs:
        return {"available": False}
    c = np.array(corrs)
    return {"available": True, "n_series": len(c), "median_corr": float(np.median(c)), "share_abs_corr_gt_0_1": float((np.abs(c) > 0.1).mean()),
            "max_abs_corr": float(np.abs(c).max())}


def volume_classes(df: pd.DataFrame) -> pd.DataFrame:
    """ABC classes by cumulative volume: A = first 80% of volume, B = next 15%, C = last 5%."""
    v = df.groupby("series_id")["sales"].sum().sort_values(ascending=False)
    cum = v.cumsum() / v.sum()
    cls = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
    # the series that crosses a threshold belongs to the higher class
    cls[0] = "A"
    return pd.DataFrame({"series_id": v.index, "volume": v.values, "cum_share": cum.values, "abc": cls})


def autocorrelation(df: pd.DataFrame, lags=(1, 2, 7, 14, 28)) -> pd.DataFrame:
    a = aggregate_series(df)
    return pd.DataFrame({"lag": list(lags), "acf": [float(a.autocorr(l)) for l in lags]})


def run_eda(df: pd.DataFrame) -> dict:
    d = with_relative_sales(df)
    vc = volume_classes(df)
    return {
        "aggregate": aggregate_series(df), "weekday": weekday_profile(d), "month": month_profile(d), "events": event_effect(d),
        "price": price_response(d), "weather": weather_response(d), "classes": demand_classes(df), "abc": vc,
        "abc_summary": vc.groupby("abc").agg(series=("series_id", "count"), volume_share=("volume", lambda x: x.sum() / vc["volume"].sum())).reset_index(),
        "acf": autocorrelation(df), "relative": d,
    }
