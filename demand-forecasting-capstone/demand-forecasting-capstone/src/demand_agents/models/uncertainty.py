"""Split-conformal prediction intervals with data-type-aware scaling.

Residuals are normalised by a noise scale (count data: sqrt(y_hat + 1); continuous data such as
dollar sales: y_hat itself), then calibrated per series with shrinkage towards the global quantile.
"""
import numpy as np


def noise_scale(yhat, kind: str = "count") -> np.ndarray:
    yhat = np.asarray(yhat, float)
    return np.sqrt(yhat + 1.0) if kind == "count" else np.maximum(yhat, 1e-6)


class ConformalCalibrator:
    def __init__(self, coverage: float = 0.90, shrink_n: int = 50, kind: str = "count"):
        self.coverage = coverage
        self.shrink_n = shrink_n
        self.kind = kind
        self.q_global = 1.0
        self.k_global = 1.0
        self.q: dict = {}
        self.k: dict = {}

    def _z(self, y, yhat):
        return (np.asarray(y, float) - np.asarray(yhat, float)) / noise_scale(yhat, self.kind)

    def _quantile(self, absz: np.ndarray) -> float:
        n = len(absz)
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        return float(np.quantile(absz, level))

    def fit(self, y, yhat, groups) -> "ConformalCalibrator":
        z = self._z(y, yhat)
        groups = np.asarray(groups)
        self.q_global = self._quantile(np.abs(z))
        self.k_global = float(np.std(z))
        for g in np.unique(groups):
            zg = z[groups == g]
            w = len(zg) / (len(zg) + self.shrink_n)
            self.q[g] = w * self._quantile(np.abs(zg)) + (1 - w) * self.q_global
            self.k[g] = w * float(np.std(zg)) + (1 - w) * self.k_global
        return self

    def _lookup(self, table: dict, default: float, groups) -> np.ndarray:
        return np.array([table.get(g, default) for g in groups])

    def intervals(self, yhat, groups, scale: float = 1.0):
        yhat = np.asarray(yhat, float)
        half = self._lookup(self.q, self.q_global, groups) * scale * noise_scale(yhat, self.kind)
        return np.clip(yhat - half, 0, None), yhat + half

    def sigma(self, yhat, groups, scale: float = 1.0) -> np.ndarray:
        yhat = np.asarray(yhat, float)
        return self._lookup(self.k, self.k_global, groups) * scale * noise_scale(yhat, self.kind)


def coverage(y, lower, upper) -> float:
    y, lower, upper = map(np.asarray, (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))
