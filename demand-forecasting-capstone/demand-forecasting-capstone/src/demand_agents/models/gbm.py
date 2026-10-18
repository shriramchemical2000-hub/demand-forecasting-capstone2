"""Global demand models (one model for all series): scikit-learn HistGradientBoosting or LightGBM."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ..data.features import CATEGORICAL


class _BaseModel:
    features: list[str]
    model: object

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(self._x(X)), 0, None)

    def _x(self, X):
        return X[self.features]

    def top_drivers(self, X: pd.DataFrame, y, n: int = 6, sample: int = 3000, seed: int = 0):
        """Permutation importance on a sample: how much does MAE worsen if a feature is shuffled?"""
        idx = np.random.default_rng(seed).choice(len(X), size=min(sample, len(X)), replace=False)
        res = permutation_importance(
            self.model, self._x(X).iloc[idx], np.asarray(y)[idx],
            scoring="neg_mean_absolute_error", n_repeats=3, random_state=seed,
        )
        order = np.argsort(res.importances_mean)[::-1][:n]
        return [(self.features[i], float(res.importances_mean[i])) for i in order]


class GBMDemandModel(_BaseModel):
    def __init__(self, params: dict, features: list[str], seed: int = 42):
        self.features = features
        cat_idx = [features.index(c) for c in CATEGORICAL if c in features]
        p = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, loss="poisson")
        p.update(params or {})
        self.model = HistGradientBoostingRegressor(
            categorical_features=cat_idx or None, early_stopping=False, random_state=seed, **p
        )

    def fit(self, X: pd.DataFrame, y) -> "GBMDemandModel":
        self.model.fit(X[self.features], y)
        return self


class LightGBMDemandModel(_BaseModel):
    """Same interface; needs ``pip install lightgbm``. Select with ``forecast.model.type: lightgbm``."""

    def __init__(self, params: dict, features: list[str], seed: int = 42, continuous: bool = False):
        from lightgbm import LGBMRegressor  # imported lazily so the dependency stays optional

        self.features = features
        p = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, objective="regression" if continuous else "poisson")
        p.update({k: v for k, v in (params or {}).items() if k not in ("max_iter", "max_leaf_nodes", "loss")})
        if "max_iter" in (params or {}):
            p["n_estimators"] = params["max_iter"]
        if "max_leaf_nodes" in (params or {}):
            p["num_leaves"] = params["max_leaf_nodes"]
        self.model = LGBMRegressor(random_state=seed, verbose=-1, **p)

    def _x(self, X):
        out = X[self.features].copy()
        for c in CATEGORICAL:
            if c in out.columns:
                out[c] = out[c].astype("category")
        return out

    def fit(self, X: pd.DataFrame, y) -> "LightGBMDemandModel":
        self.model.fit(self._x(X), y)
        return self


def make_demand_model(kind: str, params: dict, features: list[str], seed: int = 42, continuous: bool = False):
    if kind in ("hist_gbm", "gbm"):
        return GBMDemandModel(params, features, seed)
    if kind == "lightgbm":
        return LightGBMDemandModel(params, features, seed, continuous)
    raise ValueError(f"Unknown model type: {kind}")
