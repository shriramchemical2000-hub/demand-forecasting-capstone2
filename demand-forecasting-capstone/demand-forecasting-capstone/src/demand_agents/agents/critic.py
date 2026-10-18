import numpy as np
import pandas as pd

from ..evaluation.metrics import bias
from ..models.uncertainty import coverage, noise_scale
from .base import BaseAgent


class CriticAgent(BaseAgent):
    name = "Critic"
    role = "Audits calibration, flags demand shocks and weak assumptions, and sets a per-day risk multiplier."

    @property
    def kind(self) -> str:
        return self.cfg["data"].get("target_type", "count")

    # ---- 1. calibration audit (can trigger a revision request) --------------------
    def review_calibration(self) -> dict:
        s, ccfg = self.state, self.cfg["critic"]
        nominal = self.cfg["uncertainty"]["nominal_coverage"]
        chk = s.forecasts[s.forecasts["split"] == "val_check"]
        cov = coverage(chk["sales"], chk["lower"], chk["upper"])
        b = bias(chk["sales"], chk["y_hat"])
        self.observe(
            f"Coverage on the held-out validation window is {cov:.1%} (target {nominal:.0%}); bias {b:+.1%}.",
            coverage=cov, nominal=nominal, bias=b,
        )
        if abs(b) > 0.05:
            self.observe(f"Forecast bias of {b:+.1%} is above 5%; flagging for the team to investigate.", bias=b)

        record = {"coverage": cov, "nominal": nominal, "interval_scale": s.interval_scale}
        s.metrics.setdefault("calibration_checks", []).append(record)

        if cov >= nominal - ccfg["coverage_tolerance"]:
            self.decide("Calibration approved: intervals are honest enough to drive safety stock.")
            return {"approved": True, "coverage": cov, "scale": s.interval_scale}

        cal = s.calibrator
        z = np.abs(cal._z(chk["sales"], chk["y_hat"]))
        q = np.maximum(cal._lookup(cal.q, cal.q_global, chk["series_id"]), 1e-6)
        scale = float(np.clip(np.quantile(z / q, nominal), 1.0, ccfg["max_interval_scale"]))
        self.request_revision(
            "Uncertainty",
            f"Intervals under-cover ({cov:.1%} < {nominal:.0%}). Please widen them to x{scale:.2f}.",
            coverage=cov, requested_scale=scale,
        )
        return {"approved": False, "coverage": cov, "scale": scale}

    # ---- 2. learn which series react to weather ---------------------------------------
    def learn_weather_sensitivity(self, clim: pd.DataFrame) -> pd.DataFrame:
        """Which series react to weather? Link between temperature anomaly and sales relative to the recent level.

        Estimated on training data only. With ``critic.pool_sensitivity_by: item`` the same item in different
        stores is pooled (more data -> more power for low-volume series); use ``none`` when series are not
        comparable items (e.g. whole-store totals).
        """
        s, ccfg = self.state, self.cfg["critic"]
        windows = self.cfg["forecast"]["rolling_windows"]
        ref_col = f"roll_mean_{max(windows)}"
        key = "item_id" if ccfg.get("pool_sensitivity_by", "item") == "item" else "series_id"
        tr = s.feats[s.feats["split"] == "train"][["series_id", "item_id", "store_id", "month", "temp", "sales", ref_col]].merge(
            clim, on=["store_id", "month"], how="left"
        )
        tr["wz"] = (tr["temp"] - tr["mean"]) / tr["std"]
        tr = tr[(tr[ref_col] > 0) & tr["wz"].notna()]
        tr["rel"] = np.log((tr["sales"] + 1.0) / (tr[ref_col] + 1.0))
        rows = []
        for k, g in tr.groupby(key):
            n = len(g)
            ok = n > 10 and g["wz"].std() > 0 and g["rel"].std() > 0
            r = float(np.corrcoef(g["wz"], g["rel"])[0, 1]) if ok else 0.0
            t = r * np.sqrt((n - 2) / max(1e-9, 1 - r**2)) if ok else 0.0
            rows.append({key: k, "corr": r, "t_stat": t, "n": n,
                         "sensitive": abs(t) > ccfg["sensitivity_t_threshold"], "direction": 1 if t > 0 else -1})
        stats = pd.DataFrame(rows, columns=[key, "corr", "t_stat", "n", "sensitive", "direction"])
        mapping = tr[["series_id"] if key == "series_id" else ["series_id", key]].drop_duplicates()
        sens = mapping.merge(stats, on=key, how="left")
        if key != "series_id":
            sens = sens.drop(columns=[key])
        s.weather_sensitivity = sens
        share = float(sens["sensitive"].mean()) if len(sens) else 0.0
        self.observe(
            f"Weather sensitivity learned from training data (pooled by {key}): {int(sens['sensitive'].sum()) if len(sens) else 0} of "
            f"{len(sens)} series ({share:.0%}) react to temperature anomalies; only those can receive a weather flag.",
            sensitive_share=share,
        )
        return sens

    # ---- 2b. learn how much each warning sign really matters -------------------------------
    def _learn_weights(self, f: pd.DataFrame, ccfg: dict) -> dict:
        """Estimate each signal's weight from out-of-sample validation errors, shrunk towards the hand-set prior.

        weight_signal = (n * observed_extra_error + n0 * prior) / (n + n0)
        where n counts validation observations in which the signal was active. A signal that never fires on validation
        keeps its prior; a signal that fires often but does not predict larger errors is pushed towards zero.
        """
        prior = dict(ccfg["risk_weights"])
        n0 = float(ccfg.get("prior_strength", 300))
        val = f[f["split"].isin(["val_cal", "val_check"])].copy()
        flags = {"event": "event_flag", "weather": "weather_flag", "price": "price_flag"}
        quiet = pd.Series(True, index=val.index)
        for col in list(flags.values()) + ["drift_flag"]:
            if col in val.columns:
                quiet &= ~val[col].fillna(False).astype(bool)
        base = val.loc[quiet, "nz"].mean()
        learned, info = dict(prior), {}
        for name, col in flags.items():
            if col not in val.columns or name not in prior:
                continue
            act = val[val[col].fillna(False).astype(bool)]
            n = len(act)
            hat = max(0.0, act["nz"].mean() / base - 1.0) if (n > 0 and base and np.isfinite(base)) else prior[name]
            learned[name] = (n * hat + n0 * prior[name]) / (n + n0)
            info[name] = {"prior": prior[name], "observed": hat if n else None, "n": n, "weight": learned[name]}
        x = val["drift_excess"]
        m = x > 0
        n = int(m.sum())
        if n > 0:
            y = val["nz"] / val["ref_nz"] - 1.0
            b = float(np.clip((x[m] * y[m]).sum() / (x[m] ** 2).sum(), 0.0, 3.0))
        else:
            b = prior["drift"]
        learned["drift"] = (n * b + n0 * prior["drift"]) / (n + n0)
        info["drift"] = {"prior": prior["drift"], "observed": b if n else None, "n": n, "weight": learned["drift"]}
        self.state.metrics["risk_weights"] = info
        txt = ", ".join(f"{k} {v['weight']:.2f} (prior {v['prior']:.2f}, n={v['n']})" for k, v in info.items())
        self.observe(f"Risk weights learned from validation errors and shrunk towards the hand-set priors: {txt}.", weights=info)
        return learned

    # ---- 3. shock detection -> risk multiplier ---------------------------------------
    def assess_risk(self):
        s, ccfg = self.state, self.cfg["critic"]
        H = self.cfg["forecast"]["horizon"]
        self.think(
            "Three signals are checked for every series-period: (1) calendar events, (2) weather far outside "
            "the training climate for series that are weather-sensitive, in the direction that raises demand, "
            "(3) recent forecast errors that are worse than during validation. Each raises the risk multiplier."
        )
        f = s.forecasts.sort_values(["series_id", "date"]).reset_index(drop=True)
        f = f.merge(s.feats[["date", "series_id", "is_event", "temp", "month"] + (["price_rel"] if "price_rel" in s.feats.columns else [])],
                    on=["date", "series_id"], how="left")

        train = s.feats[s.feats["split"] == "train"]
        clim = train.groupby(["store_id", "month"])["temp"].agg(["mean", "std"]).reset_index()
        clim["std"] = clim["std"].fillna(1.0).clip(lower=1.0)
        sens = self.learn_weather_sensitivity(clim)

        f = f.merge(clim, on=["store_id", "month"], how="left")
        f = f.merge(sens[["series_id", "sensitive", "direction"]], on="series_id", how="left")
        f["weather_z"] = (f["temp"] - f["mean"]) / f["std"]
        f["weather_flag"] = (
            f["sensitive"].fillna(False).astype(bool) & (f["weather_z"] * f["direction"].fillna(1) > ccfg["weather_z_threshold"])
        )

        # forecast-error drift using only errors that are already observable (lagged by the horizon)
        f["nz"] = (f["sales"] - f["y_hat"]).abs() / noise_scale(f["y_hat"], self.kind)
        ref = f[f["split"] == "val_cal"].groupby("series_id")["nz"].mean().rename("ref_nz")
        f = f.merge(ref, on="series_id", how="left")
        w = ccfg["drift_window_days"]
        mp = max(3, w // 2)
        f["recent_nz"] = f.groupby("series_id")["nz"].transform(lambda x: x.rolling(w, min_periods=mp).mean().shift(H))
        f["drift_ratio"] = f["recent_nz"] / f["ref_nz"]
        f["drift_flag"] = f["drift_ratio"] > ccfg["drift_ratio_threshold"]
        f["drift_excess"] = (f["drift_ratio"] - 1.0).clip(0, 1).fillna(0.0)

        f["event_flag"] = f["is_event"].fillna(0).astype(int) == 1

        # price shock: a discount deeper than anything common in training -> demand response is extrapolated
        f["price_flag"] = False
        if "price_rel" in f.columns:
            train_pr = s.feats.loc[s.feats["split"] == "train", "price_rel"].dropna()
            if len(train_pr) > 100:
                thr = float(train_pr.quantile(ccfg.get("price_quantile", 0.01)))
                f["price_flag"] = (f["price_rel"] < thr).fillna(False)
                self.observe(f"Price signal: a price below {thr:.2f} x its recent average is rarer than the "
                             f"{ccfg.get('price_quantile', 0.01):.0%} quantile of training prices and counts as a price shock.", threshold=thr)

        wts = self._learn_weights(f, ccfg) if ccfg.get("learn_weights", True) else dict(ccfg["risk_weights"])
        f["risk_multiplier"] = np.minimum(
            ccfg["max_risk_multiplier"],
            1 + wts["event"] * f["event_flag"] + wts["weather"] * f["weather_flag"]
            + wts["drift"] * f["drift_excess"] + wts.get("price", 0.0) * f["price_flag"],
        )
        s.risk = f[["date", "series_id", "event_flag", "weather_flag", "drift_flag", "price_flag", "risk_multiplier"]].copy()

        t = f[f["split"] == "test"]
        share = lambda c: float(t[c].mean())
        self.observe(
            f"Test window: {share('event_flag'):.1%} of series-periods have an event flag, {share('weather_flag'):.1%} a "
            f"weather flag, {share('drift_flag'):.1%} an error-drift flag, {share('price_flag'):.1%} a price flag; mean risk multiplier "
            f"{t['risk_multiplier'].mean():.2f}.",
            event=share("event_flag"), weather=share("weather_flag"), drift=share("drift_flag"), price=share("price_flag"),
            mean_risk=float(t["risk_multiplier"].mean()),
        )
        self.decide("Risk multipliers published to the Uncertainty agent (interval widening) and the Planner (safety stock).")
        s.send(self.name, "Uncertainty", "Risk assessment complete; widen intervals on flagged days.")
