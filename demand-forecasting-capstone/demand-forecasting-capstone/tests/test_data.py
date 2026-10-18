import numpy as np
import pandas as pd

from demand_agents.data.features import build_features
from demand_agents.data.loaders import load_m5
from demand_agents.data.splits import assign_split, make_splits
from demand_agents.data.synthetic import generate_synthetic


def test_synthetic_shapes_and_shocks():
    df = generate_synthetic(n_stores=2, n_items=5, n_days=600, test_days=56, seed=1)
    assert df["series_id"].nunique() == 10
    assert (df["sales"] >= 0).all()
    test_start = df["date"].max() - pd.Timedelta(days=55)
    # shocks only appear in the test window
    assert df.loc[df["date"] < test_start, "shock_type"].eq("").all() or \
        set(df.loc[df["date"] < test_start, "shock_type"]) <= {""}
    assert {"heatwave", "surge"} <= set(df["shock_type"])
    # heatwave temperatures exceed anything in training
    assert df.loc[df["date"] >= test_start, "temp"].max() > df.loc[df["date"] < test_start, "temp"].max()


def test_splits_are_chronological_and_disjoint():
    dates = pd.date_range("2022-01-01", periods=400)
    sp = make_splits(dates, val_days=60, test_days=30, check_days=20)
    s = assign_split(pd.Series(dates), sp)
    assert list(s.unique()) == ["train", "val_cal", "val_check", "test"]
    assert (s == "test").sum() == 30 and (s.isin(["val_cal", "val_check"])).sum() == 60


def test_features_reject_short_lags():
    df = generate_synthetic(n_stores=1, n_items=2, n_days=200, seed=2)
    try:
        build_features(df, horizon=7, lags=[3, 7], windows=[7])
    except ValueError:
        return
    raise AssertionError("short lags should raise")


def test_features_do_not_leak_future_sales():
    df = generate_synthetic(n_stores=1, n_items=2, n_days=300, seed=3)
    H = 7
    base, names = build_features(df, H, [7, 14, 28], [7, 28])
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    changed = df.copy()
    changed.loc[changed["date"] >= cutoff, "sales"] += 1000  # alter the "future"
    alt, _ = build_features(changed, H, [7, 14, 28], [7, 28])
    sales_feats = [c for c in names if c.startswith(("lag_", "roll_"))]
    safe = base["date"] < cutoff + pd.Timedelta(days=H)
    pd.testing.assert_frame_equal(base.loc[safe, sales_feats], alt.loc[safe, sales_feats])


def test_m5_loader_with_fake_files(tmp_path):
    n_days = 6
    pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, n_days + 1)],
            "date": pd.date_range("2020-01-01", periods=n_days),
            "wm_yr_wk": [11949] * 4 + [11950] * 2,
            "event_name_1": [None, "NewYear", None, None, None, None],
        }
    ).to_csv(tmp_path / "calendar.csv", index=False)
    rows = []
    for store in ["CA_1", "CA_2"]:
        for item in ["FOODS_1_001", "FOODS_1_002"]:
            rows.append({"id": f"{item}_{store}", "item_id": item, "dept_id": "FOODS_1", "cat_id": "FOODS",
                         "store_id": store, "state_id": "CA", **{f"d_{i}": i for i in range(1, n_days + 1)}})
    pd.DataFrame(rows).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(
        {"store_id": ["CA_1"] * 4, "item_id": ["FOODS_1_001"] * 2 + ["FOODS_1_002"] * 2,
         "wm_yr_wk": [11949, 11950] * 2, "sell_price": [2.0, 2.5, 4.0, 4.5]}
    ).to_csv(tmp_path / "sell_prices.csv", index=False)

    df = load_m5(tmp_path, n_stores=1, n_items=2, max_days=None)
    assert set(df["store_id"]) == {"CA_1"}
    assert len(df) == 2 * n_days
    assert df["is_event"].sum() == 2          # one event day per item
    assert df["sell_price"].notna().all()
