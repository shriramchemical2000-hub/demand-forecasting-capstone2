"""Dataset loaders. Every loader returns the canonical long format:

    date, store_id, item_id, series_id, sales, sell_price, is_event, temp
    (+ optional extra feature columns, and ground-truth shock columns for synthetic data)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic import generate_synthetic, generate_walmart_like

CANONICAL = ["date", "store_id", "item_id", "series_id", "sales", "sell_price", "is_event", "temp"]
WALMART_EXTRAS = ["fuel_price", "cpi", "unemployment"]


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Return the modelling table. If ``data.processed_path`` exists (built by ``demand-agents build``) it is used as is;
    otherwise the raw data is loaded and the weather merged on the fly."""
    processed = (cfg["data"].get("processed_path") or "")
    if processed and Path(processed).exists():
        return read_processed(processed)
    df = load_raw_dataset(cfg)
    noaa_path = (cfg["data"].get("noaa") or {}).get("path")
    if noaa_path:
        df = merge_noaa(df, noaa_path)
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def read_processed(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "shock_type" in df.columns:
        df["shock_type"] = df["shock_type"].fillna("").astype(str)
        df["is_shock"] = df["is_shock"].astype(bool) if "is_shock" in df.columns else df["shock_type"] != ""
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def load_raw_dataset(cfg: dict) -> pd.DataFrame:
    """Stage 1 -> canonical long format, no cleaning and no weather merge."""
    source = cfg["data"]["source"]
    if source == "synthetic":
        s = cfg["data"]["synthetic"]
        df = generate_synthetic(
            n_stores=s["n_stores"], n_items=s["n_items"], n_days=s["n_days"], start_date=s["start_date"],
            test_days=cfg["split"]["test_days"], seed=cfg["project"]["seed"],
            shocks=s.get("shocks", True),
        )
    elif source == "m5":
        m = cfg["data"]["m5"]
        df = load_m5(m["raw_dir"], m["n_stores"], m["n_items"], m.get("max_days"), m.get("end_offset_days", 0))
    elif source == "walmart":
        path = Path(cfg["data"]["walmart"]["path"])
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. See docs/data_guide.md for the Walmart download.")
        df = load_walmart(path)
    elif source == "walmart_synthetic":
        df = load_walmart(generate_walmart_like(seed=cfg["project"]["seed"]))
    else:
        raise ValueError(f"Unknown data source: {source}")
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def load_walmart(path_or_df) -> pd.DataFrame:
    """Weekly store-level sales (Kaggle yasserh/walmart-dataset) -> canonical long format.

    One series per store. ``sales`` is weekly dollar sales, so the inventory simulation treats one
    'unit' as one dollar of sales volume (a value-based proxy; there is no item or price detail).
    """
    raw = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    need = {"Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Walmart file is missing columns: {sorted(missing)}")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], format="%d-%m-%Y"),
            "store_id": "W" + raw["Store"].astype(int).astype(str).str.zfill(2),
            "item_id": "STORE_TOTAL",
            "sales": raw["Weekly_Sales"].astype(float),
            "sell_price": 1.0,
            "is_event": raw["Holiday_Flag"].astype(int),
            "temp": (raw["Temperature"].astype(float) - 32.0) * 5.0 / 9.0,
            "fuel_price": raw["Fuel_Price"].astype(float),
            "cpi": raw["CPI"].astype(float),
            "unemployment": raw["Unemployment"].astype(float),
        }
    )
    df["series_id"] = df["store_id"]
    return df[CANONICAL + WALMART_EXTRAS].sort_values(["series_id", "date"]).reset_index(drop=True)


def load_m5(raw_dir, n_stores: int = 3, n_items: int = 50, max_days: int | None = 1000, end_offset_days: int = 0) -> pd.DataFrame:
    """Load a pilot subset of the M5 Forecasting dataset (Kaggle: m5-forecasting-accuracy).

    Expects calendar.csv, sales_train_evaluation.csv (or sales_train_validation.csv) and
    sell_prices.csv inside ``raw_dir``. ``end_offset_days`` drops the last N days first, so the test window
    can be moved to another period (e.g. 138 -> a window ending 5 Jan 2016 with Thanksgiving and Christmas). Rows before an item's first priced week (i.e. before the
    item was launched in that store) are dropped so launch-period zeros are not mistaken for demand.
    """
    raw = Path(raw_dir)
    sales_file = next(
        (raw / n for n in ("sales_train_evaluation.csv", "sales_train_validation.csv") if (raw / n).exists()),
        None,
    )
    if sales_file is None or not (raw / "calendar.csv").exists() or not (raw / "sell_prices.csv").exists():
        raise FileNotFoundError(f"M5 files not found in {raw}. See docs/data_guide.md for download instructions.")

    sales = pd.read_csv(sales_file)
    stores = sorted(sales["store_id"].unique())[:n_stores]
    sales = sales[sales["store_id"].isin(stores)]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if end_offset_days:
        day_cols = day_cols[:-end_offset_days]
    if max_days:
        day_cols = day_cols[-max_days:]

    volume = sales.groupby("item_id")[day_cols].sum().sum(axis=1)  # keep the highest-volume items
    keep_items = volume.sort_values(ascending=False).index[:n_items]
    sales = sales[sales["item_id"].isin(keep_items)]

    long = sales.melt(id_vars=["item_id", "store_id"], value_vars=day_cols, var_name="d", value_name="sales")
    cal = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    cal["is_event"] = cal["event_name_1"].notna().astype(int)
    long = long.merge(cal[["d", "date", "wm_yr_wk", "is_event"]], on="d", how="left")

    prices = pd.read_csv(raw / "sell_prices.csv")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long = long.sort_values(["store_id", "item_id", "date"])
    grp = long.groupby(["store_id", "item_id"])
    launched = grp["sell_price"].transform(lambda s: s.notna().cummax())  # priced at least once so far
    long = long[launched].copy()
    long["sell_price"] = long.groupby(["store_id", "item_id"])["sell_price"].transform(lambda s: s.ffill())

    long["series_id"] = long["store_id"] + "_" + long["item_id"]
    long["temp"] = np.nan  # filled by merge_noaa when a weather file is configured
    return long[CANONICAL].reset_index(drop=True)


def merge_noaa(df: pd.DataFrame, path) -> pd.DataFrame:
    """Merge daily weather. CSV columns: date, temp, plus optionally store_id OR state_id.

    ``state_id`` is matched to the prefix of the M5 store id (CA_1 -> CA).
    """
    w = pd.read_csv(path, parse_dates=["date"])
    df = df.drop(columns=["temp"], errors="ignore")
    if "store_id" in w.columns:
        keys = ["date", "store_id"]
    elif "state_id" in w.columns:
        df = df.assign(state_id=df["store_id"].str.split("_").str[0])
        keys = ["date", "state_id"]
    else:
        keys = ["date"]
    out = df.merge(w[keys + ["temp"]], on=keys, how="left")
    return out.drop(columns=["state_id"], errors="ignore")
