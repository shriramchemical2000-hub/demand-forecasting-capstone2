import numpy as np


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else float("nan")


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y, yhat) -> float:
    """Signed error relative to total demand (negative = under-forecasting)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = y.sum()
    return float((yhat - y).sum() / denom) if denom > 0 else float("nan")
