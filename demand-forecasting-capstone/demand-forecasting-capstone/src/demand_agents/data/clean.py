"""Stage 2 - cleaning. Every rule is explicit, counted and logged; nothing is silently dropped.

Rules (see docs/DATA.md for the reasoning):
  * keys / sales missing           -> row dropped
  * duplicate (series, date)       -> keep the last
  * negative sales                 -> set to 0
  * non-integer sales (count data) -> rounded
  * price <= 0                     -> missing, then carried forward within the series
  * temperature outside -60..55 C  -> missing
  * weather gaps <= 3 days         -> linear interpolation; longer gaps stay missing
Deliberately NOT done: removing demand spikes or zero-sales days. Shocks are exactly what the project studies.
"""
import numpy as np
import pandas as pd

TEMP_RANGE = (-60.0, 55.0)


def _log(steps: list, step: str, n: int, note: str = ""):
    steps.append({"step": step, "rows_affected": int(n), "note": note})


def clean_panel(df: pd.DataFrame, target_type: str = "count") -> tuple[pd.DataFrame, list[dict]]:
    steps: list[dict] = []
    d = df.copy()
    d["sales"] = pd.to_numeric(d["sales"], errors="coerce").astype(float)
    n0 = len(d)

    bad = d[["date", "series_id", "sales"]].isna().any(axis=1)
    _log(steps, "drop rows with missing date / series / sales", bad.sum())
    d = d[~bad]

    dup = d.duplicated(["series_id", "date"], keep="last")
    _log(steps, "drop duplicate (series, date) rows", dup.sum(), "kept the last occurrence")
    d = d[~dup]

    neg = d["sales"] < 0
    _log(steps, "negative sales set to 0", neg.sum())
    d.loc[neg, "sales"] = 0.0

    if target_type == "count":
        frac = (d["sales"] - d["sales"].round()).abs() > 1e-9
        _log(steps, "non-integer sales rounded (count data)", frac.sum())
        d["sales"] = d["sales"].round()

    if "sell_price" in d.columns:
        badp = d["sell_price"] <= 0
        _log(steps, "price <= 0 set to missing", badp.sum())
        d.loc[badp, "sell_price"] = np.nan
        before = d["sell_price"].isna().sum()
        d = d.sort_values(["series_id", "date"])
        d["sell_price"] = d.groupby("series_id")["sell_price"].ffill()
        _log(steps, "missing price carried forward within series", before - d["sell_price"].isna().sum(),
             f"{int(d['sell_price'].isna().sum())} leading values stay missing")

    if "temp" in d.columns:
        out = (d["temp"] < TEMP_RANGE[0]) | (d["temp"] > TEMP_RANGE[1])
        _log(steps, f"temperature outside {TEMP_RANGE} set to missing", out.sum())
        d.loc[out, "temp"] = np.nan

    if "is_event" in d.columns:
        d["is_event"] = d["is_event"].fillna(0).astype(int).clip(0, 1)

    d = d.sort_values(["series_id", "date"]).reset_index(drop=True)
    _log(steps, "rows kept", len(d), f"{len(d) / max(n0, 1):.2%} of the {n0:,} input rows")
    return d, steps


def clean_weather(w: pd.DataFrame, max_gap_days: int = 3) -> tuple[pd.DataFrame, list[dict]]:
    steps: list[dict] = []
    d = w.copy()
    dup = d.duplicated(["state_id", "date"], keep="last")
    _log(steps, "drop duplicate (state, date) rows", dup.sum())
    d = d[~dup]
    out = (d["temp"] < TEMP_RANGE[0]) | (d["temp"] > TEMP_RANGE[1])
    _log(steps, f"temperature outside {TEMP_RANGE} set to missing", out.sum())
    d.loc[out, "temp"] = np.nan

    frames, filled, still_missing = [], 0, 0
    for state, g in d.groupby("state_id"):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        s = g.set_index("date")["temp"].reindex(full)
        before = int(s.isna().sum())
        s = s.interpolate(limit=max_gap_days, limit_area="inside")
        filled += before - int(s.isna().sum())
        still_missing += int(s.isna().sum())
        frames.append(pd.DataFrame({"date": full, "state_id": state, "temp": s.values}))
    _log(steps, f"gaps of <= {max_gap_days} days interpolated linearly", filled)
    _log(steps, "days still missing after interpolation", still_missing, "kept as missing, never guessed")
    return pd.concat(frames, ignore_index=True), steps
