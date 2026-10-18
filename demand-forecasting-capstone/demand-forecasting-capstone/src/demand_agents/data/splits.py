"""Chronological train / validation / test splits."""
import numpy as np
import pandas as pd


def make_splits(dates, val_days: int, test_days: int, check_days: int) -> dict:
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    n = len(dates)
    if n <= val_days + test_days + 60:
        raise ValueError("Not enough history for the requested split sizes")
    return {
        "val_start": dates[n - test_days - val_days],
        "check_start": dates[n - test_days - check_days],
        "test_start": dates[n - test_days],
        "test_end": dates[-1],
    }


def assign_split(dates: pd.Series, splits: dict) -> pd.Series:
    d = pd.to_datetime(dates)
    conditions = [
        d < splits["val_start"],
        d < splits["check_start"],
        d < splits["test_start"],
    ]
    return pd.Series(
        np.select(conditions, ["train", "val_cal", "val_check"], default="test"),
        index=dates.index,
    )
