import numpy as np
import pandas as pd

from ..evaluation.metrics import wape
from ..inventory.policies import protection_days
from ..models.uncertainty import ConformalCalibrator, coverage
from .base import BaseAgent


class UncertaintyAgent(BaseAgent):
    name = "Uncertainty"
    role = "Turns point forecasts into calibrated prediction intervals and per-forecast sigma."

    @property
    def kind(self) -> str:
        return self.cfg["data"].get("target_type", "count")

    def _apply(self):
        s = self.state
        f = s.forecasts
        lower, upper = s.calibrator.intervals(f["y_hat"], f["series_id"], s.interval_scale)
        f["lower_base"], f["upper_base"] = lower, upper
        f["lower"], f["upper"] = lower, upper          # replaced by apply_risk once the Critic has spoken
        f["sigma"] = s.calibrator.sigma(f["y_hat"], f["series_id"], s.interval_scale)
        f["aci_mult"] = 1.0

    def _psum_z(self, f):
        """Standardised error of P-period demand sums, assuming independent daily errors (z should have std ~ 1)."""
        inv = self.cfg["inventory"]
        P = protection_days(inv["lead_time_days"], inv["review_period_days"])
        f = f.sort_values(["series_id", "date"])
        g = f.groupby("series_id")
        err = (f["sales"] - f["y_hat"]).groupby(f["series_id"]).transform(lambda x: x.rolling(P).sum())
        var = (f["sigma"] ** 2).groupby(f["series_id"]).transform(lambda x: x.rolling(P).sum())
        return (err / np.sqrt(var)).dropna()

    def apply_risk(self):
        """Widen the published intervals on days the Critic flagged (half-width x risk multiplier)."""
        s = self.state
        f = s.forecasts
        m = (
            f[["date", "series_id"]]
            .merge(s.risk[["date", "series_id", "risk_multiplier"]], on=["date", "series_id"], how="left")["risk_multiplier"]
            .fillna(1.0).to_numpy()
        )
        f["lower"] = np.clip(f["y_hat"] - (f["y_hat"] - f["lower_base"]) * m, 0, None)
        f["upper"] = f["y_hat"] + (f["upper_base"] - f["y_hat"]) * m
        wide = float(((f["split"] == "test") & (m > 1)).mean())
        self.act(
            f"Widened intervals on Critic-flagged days (half-width x risk multiplier); {wide:.1%} of test series-days affected.",
            widened_share=wide,
        )
        s.send(self.name, "Planner", "Final risk-adjusted intervals published.")

    def run(self):
        s = self.state
        nominal = self.cfg["uncertainty"]["nominal_coverage"]
        self.think(
            f"Goal: {nominal:.0%} prediction intervals. Counts get noisier as demand grows, so residuals "
            "are scaled by sqrt(forecast+1) and calibrated per series with shrinkage to the global value."
        )
        cal = s.forecasts[s.forecasts["split"] == "val_cal"]
        s.calibrator = ConformalCalibrator(nominal, kind=self.kind).fit(cal["sales"], cal["y_hat"], cal["series_id"])
        s.interval_scale = 1.0
        self._apply()
        self.act(
            f"Calibrated on {len(cal):,} validation rows (the last {self.cfg['uncertainty']['check_days']} "
            "validation days are held back for the Critic).",
            global_quantile=s.calibrator.q_global,
        )
        s.send(self.name, "Critic", "Intervals ready. Please check coverage on the held-out validation window.")

    def revise(self, scale: float):
        s = self.state
        self.act(f"Critic asked for wider intervals; scaling interval width to x{scale:.2f}.", scale=scale)
        s.interval_scale = float(scale)
        self._apply()

    def finalize(self):
        s = self.state
        nominal = self.cfg["uncertainty"]["nominal_coverage"]
        val = s.forecasts[s.forecasts["split"].isin(["val_cal", "val_check"])]
        s.calibrator = ConformalCalibrator(nominal, kind=self.kind).fit(val["sales"], val["y_hat"], val["series_id"])
        self._apply()
        self.act(
            f"Re-fitted the calibrator on the full validation window ({len(val):,} rows) "
            f"and applied interval scale x{s.interval_scale:.2f}.",
            interval_scale=s.interval_scale,
        )

        ucfg = self.cfg["uncertainty"]
        if ucfg.get("psum_inflation", True):
            z = self._psum_z(s.forecasts[s.forecasts["split"].isin(["val_cal", "val_check"])])
            s.psum_inflation = float(np.clip(z.std(), 1.0, ucfg.get("max_psum_inflation", 2.0)))
            self.observe(
                f"Errors on consecutive days are correlated, so multi-period demand is more uncertain than independent daily sigmas "
                f"suggest: the standardised error of P-period sums has std {z.std():.2f} on validation. "
                f"Planner sigma will be inflated by x{s.psum_inflation:.2f}.",
                psum_z_std=float(z.std()), inflation=s.psum_inflation,
            )

        per_series = pd.DataFrame(
            [{"series_id": sid, "val_wape": wape(g["sales"], g["y_hat"])} for sid, g in val.groupby("series_id")]
        )
        per_series["reliability"] = (1 - per_series["val_wape"]).clip(0, 1)
        s.reliability = per_series
        weakest = per_series.sort_values("reliability").head(3)["series_id"].tolist()
        self.observe(
            f"Reliability scores computed per series (1 - validation WAPE). Least reliable: {', '.join(weakest)}.",
            weakest=weakest,
        )
        s.send(self.name, "Critic", "Final intervals and sigma ready for shock assessment.")

    def apply_adaptive(self):
        """Adaptive conformal updating (Gibbs & Candes style), respecting the information delay.

        Each series keeps a log-scale. When the outcome of period t - H becomes known, a miss pushes the scale up
        and a hit lets it drift down, so the intervals follow regime changes that no feature explains.
        """
        s = self.state
        acfg = self.cfg["uncertainty"].get("adaptive", {})
        if not acfg.get("enabled", True):
            return
        H = self.cfg["forecast"]["horizon"]
        alpha = 1.0 - self.cfg["uncertainty"]["nominal_coverage"]
        gamma, cap = acfg.get("gamma", 0.15), acfg.get("max_log_scale", 0.7)
        f = s.forecasts
        mask = f["split"].isin(["val_check", "test"])
        for sid, g in f[mask].groupby("series_id"):
            idx = g.sort_values("date").index.to_numpy()
            y = f.loc[idx, "sales"].to_numpy()
            yh = f.loc[idx, "y_hat"].to_numpy()
            lo0, up0 = f.loc[idx, "lower"].to_numpy(), f.loc[idx, "upper"].to_numpy()
            n = len(idx)
            lo, up, mult, miss = np.empty(n), np.empty(n), np.ones(n), np.zeros(n)
            log_s = 0.0
            for i in range(n):
                if i >= H:
                    log_s = float(np.clip(log_s + gamma * (miss[i - H] - alpha), -0.3, cap))
                m = np.exp(log_s)
                mult[i] = m
                lo[i] = max(0.0, yh[i] - (yh[i] - lo0[i]) * m)
                up[i] = yh[i] + (up0[i] - yh[i]) * m
                miss[i] = float(y[i] < lo[i] or y[i] > up[i])
            f.loc[idx, "lower"], f.loc[idx, "upper"], f.loc[idx, "aci_mult"] = lo, up, mult
        t = f[f["split"] == "test"]
        self.act(
            f"Adaptive conformal updating applied (gamma={gamma}); mean test interval multiplier {t['aci_mult'].mean():.2f}, "
            f"max {t['aci_mult'].max():.2f}.",
            gamma=gamma,
        )
