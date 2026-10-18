import numpy as np

from ..data.features import build_features
from ..data.splits import assign_split, make_splits
from ..evaluation.metrics import wape
from ..models.baselines import rolling_classical_forecast, seasonal_naive
from ..models.gbm import make_demand_model
from .base import BaseAgent


class ForecasterAgent(BaseAgent):
    name = "Forecaster"
    role = "Predicts demand one forecast-horizon ahead per series; picks the best candidate on held-out validation data."

    def run(self):
        s, fc = self.state, self.cfg["forecast"]
        dcfg = self.cfg["data"]
        H = fc["horizon"]
        self.think(
            f"Goal: forecast demand {H} periods ahead for {s.panel['series_id'].nunique()} series. "
            "Every sales-derived feature must be at least that old, otherwise the evaluation leaks the future."
        )

        feats, names = build_features(
            s.panel, H, fc["lags"], fc["rolling_windows"], freq=dcfg.get("frequency", "D"),
            extra_features=dcfg.get("extra_features", []), price_window=fc.get("price_window", 56),
        )
        s.splits = make_splits(
            feats["date"], self.cfg["split"]["val_days"], self.cfg["split"]["test_days"],
            self.cfg["uncertainty"]["check_days"],
        )
        feats["split"] = assign_split(feats["date"], s.splits)
        feats["seasonal_naive"] = seasonal_naive(feats, H, dcfg.get("season_period", 7))
        s.feats, s.features = feats, names
        self.act(
            f"Built {len(names)} features; chronological split train < {s.splits['val_start'].date()}, "
            f"validation to {s.splits['test_start'].date()}, test to {s.splits['test_end'].date()}.",
            features=names,
        )

        max_lag = max(fc["lags"])
        train = feats[feats["split"] == "train"].dropna(subset=[f"lag_{max_lag}"])
        mtype = fc["model"].get("type", "hist_gbm")
        model = make_demand_model(mtype, fc["model"]["params"], names, seed=self.cfg["project"]["seed"],
                                  continuous=dcfg.get("target_type", "count") == "continuous")
        model.fit(train, train["sales"])
        s.model = model
        self.act(f"Trained {mtype} model on {len(train):,} training rows.", params=fc["model"]["params"])

        ev = feats[feats["split"] != "train"].copy()
        ev["gbm"] = model.predict(ev)
        val = ev[ev["split"].str.startswith("val")]
        drivers = model.top_drivers(val, val["sales"])
        self.observe("Top demand drivers (permutation importance): " + ", ".join(n for n, _ in drivers), drivers=drivers)

        # ---- candidate selection: gradient boosting vs a blend with classical ETS vs seasonal naive ----------------
        sel = fc.get("selection", {})
        if sel.get("blend_with_ets", True) or "ets" in fc["baselines"]:
            self.act("Running the classical ETS baseline (rolling origin) over validation + test; it is both a benchmark and a blend candidate.")
            ets, sigma = rolling_classical_forecast(
                s.panel, s.splits["val_start"], H, method="ets", window=fc.get("classical_window", 365),
                season_period=dcfg.get("season_period", 7), sigma_at=s.splits["test_start"],
            )
            s.baselines["ets"], s.baseline_sigma["ets"] = ets, sigma
            ev = ev.merge(ets.rename(columns={"y_hat": "ets"}), on=["date", "series_id"], how="left")
        else:
            ev["ets"] = np.nan

        cal = ev[ev["split"] == "val_cal"].dropna(subset=["ets"])
        chk = ev[ev["split"] == "val_check"]
        w_best = 1.0
        if sel.get("blend_with_ets", True) and len(cal) > 0:
            grid = np.linspace(0, 1, 11)
            w_best = float(grid[np.argmin([wape(cal["sales"], w * cal["gbm"] + (1 - w) * cal["ets"]) for w in grid])])
        ev["blend"] = w_best * ev["gbm"] + (1 - w_best) * ev["ets"].fillna(ev["gbm"])

        scores = {
            "gbm": wape(chk["sales"], chk["gbm"]),
            "blend": wape(chk["sales"], ev.loc[chk.index, "blend"]),
            "seasonal_naive": wape(chk["sales"], chk["seasonal_naive"].fillna(chk["gbm"])),
        }
        if "ets" in chk.columns and chk["ets"].notna().any():
            scores["ets"] = wape(chk.dropna(subset=["ets"])["sales"], chk.dropna(subset=["ets"])["ets"])
        self.observe(
            "Held-out validation WAPE: " + ", ".join(f"{k} {v:.3f}" for k, v in scores.items())
            + f"; blend weight on the model chosen on the calibration window = {w_best:.1f}.",
            scores=scores, blend_weight=w_best,
        )
        gain = sel.get("min_relative_gain", 0.02)  # prefer the simpler model unless the blend is clearly better
        adjusted = dict(scores)
        adjusted["blend"] = scores["blend"] / (1 - gain) if w_best < 1.0 else scores["blend"]
        choice = min(("gbm", "blend", "seasonal_naive"), key=lambda k: adjusted[k])
        ev["y_hat"] = ev["seasonal_naive"].fillna(ev["gbm"]) if choice == "seasonal_naive" else ev[choice]
        s.metrics["forecaster"] = {"choice": choice, "blend_weight": w_best, "scores": scores}
        self.decide(
            f"Using '{choice}' (validation WAPE {scores[choice]:.3f}; "
            f"{(1 - scores[choice] / scores['seasonal_naive']):.0%} lower than seasonal naive)."
        )

        keep = ["date", "series_id", "store_id", "item_id", "split", "sales", "y_hat", "gbm", "seasonal_naive"]
        s.forecasts = ev[keep].reset_index(drop=True)
        s.send(self.name, "Uncertainty", f"{len(s.forecasts):,} forecasts ready (validation + test).")
