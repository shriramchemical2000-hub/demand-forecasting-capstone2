"""Stage 3 - integration: raw -> clean -> audited -> merged modelling table.

Which datasets are combined, and which are not (see docs/DATA.md):
  * M5 sales + M5 calendar + M5 prices     -> already one table inside the M5 loader
  * M5 + NOAA weather                       -> merged here by state and date
  * Walmart                                 -> kept SEPARATE (weekly dollar sales, other stores, other period, no shared keys)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import read_noaa_raw
from .audit import audit_panel, audit_weather
from .clean import clean_panel, clean_weather
from .loaders import load_raw_dataset


def merge_weather_frame(df: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Merge a daily weather table (date, temp + store_id OR state_id) into the panel. State ids match the store-id prefix (CA_1 -> CA)."""
    d = df.drop(columns=["temp"], errors="ignore")
    if "store_id" in w.columns:
        keys = ["date", "store_id"]
    elif "state_id" in w.columns:
        d = d.assign(state_id=d["store_id"].str.split("_").str[0])
        keys = ["date", "state_id"]
    else:
        keys = ["date"]
    out = d.merge(w[keys + ["temp"]], on=keys, how="left")
    return out.drop(columns=["state_id"], errors="ignore")


def load_weather(cfg: dict) -> pd.DataFrame | None:
    noaa = (cfg["data"].get("noaa") or {})
    raw_dir, path = noaa.get("raw_dir"), noaa.get("path")
    if raw_dir and Path(raw_dir).exists() and list(Path(raw_dir).glob("*.csv")):
        return read_noaa_raw(raw_dir)
    if path and Path(path).exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


def build_dataset(cfg: dict, interim_dir="data/interim", processed_dir="data/processed", reports_dir="reports", write: bool = True) -> dict:
    name = cfg["data"]["source"]
    freq = cfg["data"].get("frequency", "D")
    kind = cfg["data"].get("target_type", "count")
    season = cfg["data"].get("season_period", 7)

    raw = load_raw_dataset(cfg)
    audit_before = audit_panel(raw, freq, name, season)
    clean, log = clean_panel(raw, kind)

    weather_report, merged, weather_clean, wlog = None, clean, None, []
    w = load_weather(cfg)
    if w is not None:
        weather_clean, wlog = clean_weather(w)
        states = sorted(clean["store_id"].str.split("_").str[0].unique()) if name == "m5" else None
        weather_report = audit_weather(weather_clean, clean["date"].unique(), states)
        merged = merge_weather_frame(clean, weather_clean)
        weather_report["panel_rows_with_temperature"] = float(merged["temp"].notna().mean())

    audit_after = audit_panel(merged, freq, name, season)
    summary = {"name": name, "cleaning_log": log, "weather_cleaning_log": wlog, "audit_raw": audit_before,
               "audit_final": audit_after, "weather_audit": weather_report, "rows_final": int(len(merged))}

    if write:
        interim, processed, reports = Path(interim_dir), Path(processed_dir), Path(reports_dir)
        for d in (interim, processed, reports / "tables"):
            d.mkdir(parents=True, exist_ok=True)
        clean.to_csv(interim / f"{name}_clean.csv.gz", index=False)
        if weather_clean is not None:
            weather_clean.to_csv(interim / "weather_clean.csv", index=False)
        out_path = Path(cfg["data"].get("processed_path") or processed / f"{name}_model_table.csv.gz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        summary["processed_path"] = str(out_path)
        pd.DataFrame(log).to_csv(reports / "tables" / f"cleaning_log_{name}.csv", index=False)
        import json
        (reports / "tables" / f"data_audit_{name}.json").write_text(json.dumps(summary, indent=2, default=str))
    summary["table"] = merged
    return summary
