import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from demand_agents.config import load_config
from demand_agents.data.loaders import load_dataset, load_m5, load_walmart, merge_noaa
from demand_agents.data.synthetic import generate_walmart_like
from demand_agents.orchestrator import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_config_inheritance():
    cfg = load_config(ROOT / "configs" / "walmart.yaml")
    assert cfg["data"]["frequency"] == "W" and cfg["forecast"]["horizon"] == 2
    assert cfg["critic"]["coverage_tolerance"] == 0.05            # inherited from default.yaml
    assert cfg["inventory"]["nowcast"]["window"] == 2 and cfg["inventory"]["nowcast"]["enabled"] is False


def test_walmart_loader_maps_real_file_format(tmp_path):
    raw = generate_walmart_like(n_stores=3, n_weeks=60)
    path = tmp_path / "Walmart.csv"
    raw.to_csv(path, index=False)
    df = load_walmart(path)
    assert df["series_id"].nunique() == 3 and len(df) == 180
    assert set(["fuel_price", "cpi", "unemployment"]) <= set(df.columns)
    first = raw.iloc[0]
    assert abs(df.iloc[0]["temp"] - (first["Temperature"] - 32) * 5 / 9) < 1e-9   # Fahrenheit -> Celsius
    assert all(g["date"].is_monotonic_increasing for _, g in df.groupby("series_id"))


def test_weekly_continuous_pipeline_end_to_end(tmp_path):
    cfg = load_config(ROOT / "configs" / "walmart.yaml", {"data.source": "walmart_synthetic"})
    cfg["output_dir"] = str(tmp_path / "w")
    cfg["forecast"]["baselines"] = ["seasonal_naive"]
    cfg["forecast"]["selection"] = {"blend_with_ets": False}
    cfg["forecast"]["model"]["params"]["max_iter"] = 40
    state = run_pipeline(cfg, verbose=False)
    inv = state.inventory_summary.set_index("policy")
    assert {"agentic", "gbm_static", "gbm_calibrated"} <= set(inv.index)
    assert ((inv["fill_rate"] > 0.5) & (inv["fill_rate"] <= 1)).all()
    f = state.forecasts
    assert (f["upper"] >= f["y_hat"]).all() and (f["lower"] <= f["y_hat"]).all()
    assert (tmp_path / "w" / "traces.jsonl").exists()


def _fake_m5(tmp_path, n_days=8):
    pd.DataFrame(
        {"d": [f"d_{i}" for i in range(1, n_days + 1)], "date": pd.date_range("2020-01-01", periods=n_days),
         "wm_yr_wk": [11949] * 4 + [11950] * 4, "event_name_1": [None] * n_days}
    ).to_csv(tmp_path / "calendar.csv", index=False)
    rows = [{"id": "x", "item_id": "FOODS_1_001", "dept_id": "FOODS_1", "cat_id": "FOODS", "store_id": "CA_1",
             "state_id": "CA", **{f"d_{i}": 0 if i <= 4 else 5 for i in range(1, n_days + 1)}}]
    pd.DataFrame(rows).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    # the item only has a price from week 11950 on -> not launched during the first 4 days
    pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1_001"], "wm_yr_wk": [11950], "sell_price": [3.0]}).to_csv(
        tmp_path / "sell_prices.csv", index=False
    )


def test_m5_loader_drops_pre_launch_rows(tmp_path):
    _fake_m5(tmp_path)
    df = load_m5(tmp_path, n_stores=1, n_items=1, max_days=None)
    assert len(df) == 4                                   # the 4 unpriced pre-launch days are removed
    assert (df["sales"] == 5).all() and (df["sell_price"] == 3.0).all()


def test_noaa_merge_by_state_prefix(tmp_path):
    _fake_m5(tmp_path)
    df = load_m5(tmp_path, n_stores=1, n_items=1, max_days=None)
    w = pd.DataFrame({"date": pd.date_range("2020-01-05", periods=4), "state_id": "CA", "temp": [10.0, 11, 12, 13]})
    w.to_csv(tmp_path / "w.csv", index=False)
    merged = merge_noaa(df, tmp_path / "w.csv")
    assert merged["temp"].tolist() == [10.0, 11.0, 12.0, 13.0] and "state_id" not in merged.columns
def test_m5_end_offset_moves_the_window(tmp_path):
    _fake_m5(tmp_path)
    full = load_m5(tmp_path, n_stores=1, n_items=1, max_days=None)
    early = load_m5(tmp_path, n_stores=1, n_items=1, max_days=None, end_offset_days=2)
    assert early["date"].max() == full["date"].max() - pd.Timedelta(days=2)
    assert len(early) == len(full) - 2
