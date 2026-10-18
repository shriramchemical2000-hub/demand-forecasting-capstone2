"""Stage 2b - data audit: measure the quality of a dataset *before* modelling and say what to do about each finding."""
import numpy as np
import pandas as pd

ADI_CUT, CV2_CUT = 1.32, 0.49  # Syntetos-Boylan demand classification thresholds


def _issue(issues, severity, check, detail, action):
    issues.append({"severity": severity, "check": check, "detail": detail, "action": action})


def demand_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Per series: average demand interval (ADI), squared coefficient of variation of non-zero demand (CV2) and its class."""
    rows = []
    for sid, g in df.groupby("series_id"):
        y = g["sales"].to_numpy(float)
        nz = y[y > 0]
        if len(nz) == 0:
            rows.append({"series_id": sid, "adi": np.inf, "cv2": np.nan, "class": "no demand"})
            continue
        adi = len(y) / len(nz)
        cv2 = (nz.std() / nz.mean()) ** 2 if nz.mean() > 0 else np.nan
        cls = ("smooth" if cv2 < CV2_CUT else "erratic") if adi < ADI_CUT else ("intermittent" if cv2 < CV2_CUT else "lumpy")
        rows.append({"series_id": sid, "adi": adi, "cv2": cv2, "class": cls})
    return pd.DataFrame(rows)


def mass_zero_days(df: pd.DataFrame, share: float = 0.9, min_series: int = 5) -> list[str]:
    """Dates on which almost every series sold nothing (store closures, e.g. Christmas Day)."""
    if df["series_id"].nunique() < min_series:
        return []
    z = (df["sales"] == 0).groupby(df["date"]).mean()
    return [str(d.date()) for d in z[z >= share].index]


def audit_panel(df: pd.DataFrame, freq: str = "D", name: str = "dataset", season_period: int = 7) -> dict:
    issues: list[dict] = []
    step = pd.Timedelta(days=1 if freq == "D" else 7)
    n_series = df["series_id"].nunique()
    dmin, dmax = df["date"].min(), df["date"].max()

    # ---- structure ----
    dup = int(df.duplicated(["series_id", "date"]).sum())
    g = df.groupby("series_id")["date"].agg(["min", "max", "nunique"]).rename(columns={"nunique": "count"})  # unique dates: duplicates must not hide gaps
    expected = ((g["max"] - g["min"]) / step).astype(int) + 1
    gaps = (expected - g["count"]).clip(lower=0)
    late = int((g["min"] > dmin + 0.05 * (dmax - dmin)).sum())
    early_end = int((g["max"] < dmax - 0.05 * (dmax - dmin)).sum())
    short = int((g["count"] < 3 * season_period).sum())
    if dup:
        _issue(issues, "error", "duplicate rows", f"{dup:,} duplicate (series, date) rows", "drop, keep the last")
    if (gaps > 0).any():
        _issue(issues, "warn", "missing periods", f"{int((gaps > 0).sum())} series have gaps ({int(gaps.sum()):,} periods in total)",
               "do not fill with zero (not the same as no demand); models see NaN lags")
    if late:
        _issue(issues, "info", "late-starting series", f"{late} series start after the first 5% of the period (launched late)",
               "pre-launch rows are removed so launch zeros are not read as demand")
    if short:
        _issue(issues, "warn", "very short series", f"{short} series have fewer than {3 * season_period} periods", "forecasts for them are unreliable")

    # ---- sales ----
    neg = int((df["sales"] < 0).sum())
    nan_sales = int(df["sales"].isna().sum())
    zero_share = float((df["sales"] == 0).mean())
    per_series_zero = (df["sales"] == 0).groupby(df["series_id"]).mean()
    if neg:
        _issue(issues, "error", "negative sales", f"{neg:,} rows", "set to 0 (returns are not modelled)")
    if nan_sales:
        _issue(issues, "error", "missing sales", f"{nan_sales:,} rows", "drop the rows")
    classes = demand_classes(df)
    class_counts = classes["class"].value_counts().to_dict()
    n_hard = class_counts.get("intermittent", 0) + class_counts.get("lumpy", 0)
    if n_hard:
        _issue(issues, "info", "intermittent demand", f"{n_hard} of {n_series} series are intermittent or lumpy; zero share overall {zero_share:.1%}",
               "use count losses (Poisson), judge with WAPE not MAPE, expect weak classical baselines")
    if zero_share > 0.5:
        _issue(issues, "warn", "mostly zeros", f"{zero_share:.0%} of observations are zero", "consider aggregating or dropping very slow items")

    closed = mass_zero_days(df)
    if closed:
        _issue(issues, "info", "store-closure days", f"{len(closed)} dates where >= 90% of series sold nothing: {', '.join(closed[:6])}",
               "keep them; make sure they are marked as events so they are known in advance")

    # ---- outliers (kept, flagged) ----
    med = df.groupby("series_id")["sales"].transform("median")
    mad = (df["sales"] - med).abs().groupby(df["series_id"]).transform("median")
    robust_z = (df["sales"] - med).abs() / (1.4826 * mad + 1e-9)
    big = (robust_z > 6) & (mad > 0)
    out_share = float(big.mean())
    if out_share > 0:
        _issue(issues, "info", "demand spikes", f"{out_share:.2%} of observations are > 6 robust sigmas from their series median",
               "KEPT: spikes are the shocks the Critic must detect; they are not errors")

    # ---- price ----
    price = {"present": "sell_price" in df.columns}
    if price["present"]:
        p = df["sell_price"]
        price.update({"missing_share": float(p.isna().mean()), "non_positive": int((p <= 0).sum()),
                      "constant_series_share": float((df.groupby("series_id")["sell_price"].nunique() <= 1).mean())})
        if price["missing_share"] > 0:
            _issue(issues, "warn", "missing prices", f"{price['missing_share']:.1%} of rows", "carry forward within series; leading gaps stay missing")
        if price["non_positive"]:
            _issue(issues, "error", "non-positive prices", f"{price['non_positive']:,} rows", "set to missing")
        if price["constant_series_share"] > 0.95:
            _issue(issues, "info", "no price variation", "price is constant for almost every series", "price features are skipped automatically")

    # ---- weather ----
    weather = {"present": "temp" in df.columns}
    if weather["present"]:
        t = df["temp"]
        weather.update({"missing_share": float(t.isna().mean()), "min": float(t.min()) if t.notna().any() else None,
                        "max": float(t.max()) if t.notna().any() else None})
        if weather["missing_share"] == 1.0:
            _issue(issues, "warn", "no weather", "temperature is missing for every row", "weather features and the Critic's weather signal are inactive; merge NOAA")
        elif weather["missing_share"] > 0:
            _issue(issues, "warn", "partial weather", f"{weather['missing_share']:.1%} of rows have no temperature", "trees accept NaN; check the NOAA coverage")

    events = {"present": "is_event" in df.columns}
    if events["present"]:
        events["event_day_share"] = float(df.drop_duplicates("date")["is_event"].mean())

    return {
        "name": name, "frequency": freq,
        "overview": {"rows": int(len(df)), "series": int(n_series), "start": str(dmin.date()), "end": str(dmax.date()),
                     "periods": int(df["date"].nunique()), "stores": int(df["store_id"].nunique()) if "store_id" in df.columns else None,
                     "items": int(df["item_id"].nunique()) if "item_id" in df.columns else None},
        "quality": {"duplicates": dup, "series_with_gaps": int((gaps > 0).sum()), "missing_periods": int(gaps.sum()),
                    "late_start_series": late, "early_end_series": early_end, "short_series": short,
                    "negative_sales": neg, "missing_sales": nan_sales},
        "demand": {"zero_share": zero_share, "series_zero_share_median": float(per_series_zero.median()),
                   "series_zero_share_p90": float(per_series_zero.quantile(0.9)), "classes": class_counts,
                   "outlier_share": out_share, "mass_zero_days": closed,
                   "mean_sales": float(df["sales"].mean()), "median_sales": float(df["sales"].median())},
        "price": price, "weather": weather, "events": events, "issues": issues,
    }


def audit_weather(w: pd.DataFrame, panel_dates=None, states=None) -> dict:
    issues: list[dict] = []
    out = {"rows": int(len(w)), "states": sorted(w["state_id"].unique().tolist()),
           "start": str(w["date"].min().date()), "end": str(w["date"].max().date()),
           "missing_share": float(w["temp"].isna().mean()),
           "temp_min": float(w["temp"].min()), "temp_max": float(w["temp"].max())}
    if out["missing_share"] > 0.02:
        _issue(issues, "warn", "weather gaps", f"{out['missing_share']:.1%} of station-days have no temperature", "interpolate gaps <= 3 days only")
    if panel_dates is not None:
        pd_ = pd.DatetimeIndex(panel_dates).unique()
        covered = pd_.isin(w["date"].unique()).mean()
        out["panel_date_coverage"] = float(covered)
        if covered < 0.99:
            _issue(issues, "warn", "weather does not cover the sales period", f"only {covered:.1%} of sales dates have weather",
                   "extend the NOAA download range")
    if states is not None:
        miss = sorted(set(states) - set(w["state_id"].unique()))
        out["states_without_weather"] = miss
        if miss:
            _issue(issues, "warn", "missing states", f"no weather for {miss}", "add stations for these states")
    out["issues"] = issues
    return out
