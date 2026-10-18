import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import periods_per_year
from ..inventory.policies import base_stock_components, nowcast_factor, protection_days
from ..inventory.simulator import simulate_base_stock, summarize
from .base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "Planner"
    role = "Converts forecasts + uncertainty + risk into order-up-to levels and tests them in a simulation."

    def run(self):
        s, inv = self.state, self.cfg["inventory"]
        H = self.cfg["forecast"]["horizon"]
        L, R = inv["lead_time_days"], inv["review_period_days"]
        P = protection_days(L, R) + int(inv.get("lead_time_buffer", 0))
        if P > H:
            raise ValueError(f"protection interval ({P} periods) exceeds the forecast horizon ({H} periods)")
        z = float(norm.ppf(inv["service_level"]))
        nc = inv.get("nowcast", {})
        self.think(
            f"{inv.get('order_rule', 'base_stock')} policy: order-up-to = expected demand over {P} periods (lead time {L} + review {R}"
            f"{' + buffer ' + str(inv.get('lead_time_buffer')) if inv.get('lead_time_buffer') else ''}) plus "
            f"z={z:.2f} x forecast std ({inv['service_level']:.0%} target). Five policies are compared on the same test "
            "window as a ladder: classical -> better forecast -> calibrated uncertainty -> Critic risk -> nowcast "
            "correction, so each agent's contribution can be separated."
        )

        f = s.forecasts[s.forecasts["split"] == "test"].merge(s.risk, on=["date", "series_id"], how="left")
        series = sorted(f["series_id"].unique())
        dates = sorted(f["date"].unique())

        def mat(df, col, idx=dates):
            return df.pivot(index="date", columns="series_id", values=col).reindex(index=idx, columns=series).to_numpy(float)

        demand = mat(f, "sales")
        y_hat = mat(f, "y_hat")
        ucfg = self.cfg["uncertainty"]
        sigma = mat(f, "sigma")
        if ucfg.get("psum_inflation", True):
            sigma = sigma * s.psum_inflation
        if ucfg.get("adaptive", {}).get("use_in_planner", True) and "aci_mult" in f.columns:
            sigma = sigma * mat(f, "aci_mult")
        risk = np.nan_to_num(mat(f, "risk_multiplier"), nan=1.0)

        # static (textbook) safety-stock sigma: one number per series from validation residuals
        val = s.forecasts[s.forecasts["split"].isin(["val_cal", "val_check"])].copy()
        val["resid"] = val["sales"] - val["y_hat"]
        static = val.groupby("series_id")["resid"].std().reindex(series).fillna(val["resid"].std()).to_numpy()
        static_sigma = np.tile(static, (len(dates), 1))

        if "ets" in s.baselines:
            b = s.baselines["ets"]
            classical_name = "classical_ets"
            classical_fc = mat(b, "y_hat")
            fallback_sd = float(np.nanmean(static))
            sd_vec = np.array([s.baseline_sigma["ets"].get(x, fallback_sd) for x in series], dtype=float)
            bad_sd = ~(np.isfinite(sd_vec) & (sd_vec > 0))
            sd_vec = np.where(bad_sd, np.where(np.isfinite(static) & (static > 0), static, fallback_sd), sd_vec)
            classical_sigma = np.tile(sd_vec, (len(dates), 1))
            if bad_sd.any():
                self.observe(f"{int(bad_sd.sum())} series had an unusable ETS sigma; the validation-residual sigma was used for them.")
        else:
            classical_name = "classical_seasonal_naive"
            classical_fc = mat(f, "seasonal_naive")
            classical_sigma = static_sigma
        bad_fc = ~np.isfinite(classical_fc)
        if bad_fc.any():
            self.observe(f"{int(bad_fc.sum())} classical forecasts were missing or unusable and were replaced by the model forecast "
                         "(this slightly favours the classical policy).")
        classical_fc = np.where(bad_fc, y_hat, classical_fc)
        classical_sigma = np.where(np.isfinite(classical_sigma) & (classical_sigma > 0), classical_sigma, static_sigma)

        # nowcast: re-centre expected demand when recently *observed* demand is significantly off the forecast
        adjust = None
        if nc.get("enabled", True):
            K = int(nc.get("window", 7))
            hist = s.forecasts[s.forecasts["split"].isin(["val_check", "test"])]
            hist_dates = sorted(hist["date"].unique())
            ext = hist_dates[-(len(dates) + K - 1):]
            ignore = None
            if nc.get("ignore_event_periods", True) and "event_flag" in s.risk.columns:
                ignore = np.nan_to_num(mat(s.risk, "event_flag", ext), nan=0.0) > 0.5
            adjust = nowcast_factor(
                mat(hist, "sales", ext), mat(hist, "y_hat", ext), mat(hist, "sigma", ext), K,
                nc.get("z_threshold", 2.0), nc.get("shrink", 0.7), tuple(nc.get("clip", (0.8, 2.0))), ignore,
            )

        policies = {
            classical_name: (classical_fc, classical_sigma, None, None),
            "gbm_static": (y_hat, static_sigma, None, None),
            "gbm_calibrated": (y_hat, sigma, None, None),
            "agentic_risk": (y_hat, sigma, risk, None),
            "agentic": (y_hat, sigma, risk, adjust),
        }
        if adjust is None:
            policies.pop("agentic_risk")  # identical to "agentic" when the nowcast is disabled
        abc = inv.get("abc_service_levels", {})
        z_vec = None
        if abc.get("enabled"):
            volume = s.panel[s.panel["date"] < dates[0]].groupby("series_id")["sales"].sum().reindex(series).fillna(0.0)
            rank = volume.rank(ascending=False, method="first")
            cls = np.where(rank <= 0.2 * len(series), "A", np.where(rank <= 0.5 * len(series), "B", "C"))
            z_vec = norm.ppf(np.array([abc[c] for c in cls]))
            policies["agentic_abc"] = (y_hat, sigma, risk, adjust)
            self.observe(
                f"Per-item service levels by volume class: {int((cls == 'A').sum())} A-items at {abc['A']:.0%}, "
                f"{int((cls == 'B').sum())} B at {abc['B']:.0%}, {int((cls == 'C').sum())} C at {abc['C']:.0%}.",
            )

        price = (
            s.panel[s.panel["date"].isin(dates)].groupby("series_id")["sell_price"].median().reindex(series).fillna(1.0).to_numpy()
        )
        s.price = pd.Series(price, index=series)
        holding = inv["holding_rate_annual"] / periods_per_year(self.cfg) * price
        stockout = inv["stockout_cost_frac_of_price"] * price

        rows, daily, comps_agentic = [], [], None
        for name, (fc, sg, rk, adj) in policies.items():
            zz = z_vec if (name == "agentic_abc" and z_vec is not None) else z
            mean_d, safety = base_stock_components(fc, sg, zz, P, rk, adj)
            S = mean_d + safety
            rop = None
            if inv.get("order_rule", "base_stock") == "s_S":
                rop = S - inv.get("s_S_gap_periods", 2) * mean_d / P
            sim = simulate_base_stock(
                demand, S, L, reorder_point=rop, batch_size=inv.get("batch_size", 1),
                min_order_qty=inv.get("min_order_qty", 0), delay_prob=inv.get("lead_time_delay_prob", 0.0),
                delay_max=inv.get("lead_time_delay_max", 0), rng=np.random.default_rng(self.cfg["project"]["seed"]),
            )
            summ = summarize(sim, demand, holding, stockout, inv.get("fixed_order_cost", 0.0))
            if not all(np.isfinite(v) for v in summ.values()):
                self.observe(f"WARNING: policy '{name}' produced non-finite metrics; check the inputs for missing values.", policy=name)
                import warnings
                warnings.warn(f"policy '{name}' produced non-finite metrics")
            rows.append({"policy": name, **summ})
            self.observe(
                f"{name}: fill rate {summ['fill_rate']:.1%}, stockout rate {summ['stockout_rate']:.1%}, "
                f"avg stock {summ['avg_on_hand_units']:.1f} units, total cost {summ['total_cost']:.1f}.",
                policy=name, **summ,
            )
            T, N = demand.shape
            daily.append(
                pd.DataFrame(
                    {
                        "policy": name,
                        "date": np.repeat(dates, N),
                        "series_id": np.tile(series, T),
                        "demand": demand.ravel(),
                        "sold": sim.sold.ravel(),
                        "lost": sim.lost.ravel(),
                        "on_hand_end": sim.on_hand_end.ravel(),
                        "order_qty": sim.orders.ravel(),
                        "order_up_to": S[1:].ravel(),
                    }
                )
            )
            if name == "agentic":
                comps_agentic = (mean_d, safety)

        s.inventory_summary = pd.DataFrame(rows)
        s.inventory_daily = pd.concat(daily, ignore_index=True)

        # recommendation for the next planning cycle (state at the end of the test window)
        mean_d, safety = comps_agentic
        last = f["date"].max()
        flags = s.risk[s.risk["date"] == last].set_index("series_id")
        rel = s.reliability.set_index("series_id")["reliability"] if s.reliability is not None else pd.Series(dtype=float)
        recs = []
        for j, sid in enumerate(series):
            reasons = []
            if sid in flags.index:
                fl = flags.loc[sid]
                if fl["event_flag"]:
                    reasons.append("calendar event")
                if fl["weather_flag"]:
                    reasons.append("abnormal weather")
                if fl["drift_flag"]:
                    reasons.append("recent forecast errors above normal")
            text = f"Cover {P} periods of expected demand ({mean_d[-1, j]:,.0f} units) plus {safety[-1, j]:,.0f} units safety stock"
            if adjust is not None and abs(adjust[-1, j] - 1) > 1e-9:
                text += f"; expected demand re-centred x{adjust[-1, j]:.2f} because recent sales deviate from forecast"
            text += f"; safety stock raised x{risk[-1, j]:.2f} because of {', '.join(reasons)}." if reasons else "; no shock signals."
            recs.append(
                {
                    "series_id": sid,
                    "as_of_date": last,
                    "expected_demand_protection_period": mean_d[-1, j],
                    "safety_stock": safety[-1, j],
                    "order_up_to": mean_d[-1, j] + safety[-1, j],
                    "risk_multiplier": risk[-1, j],
                    "nowcast_factor": adjust[-1, j] if adjust is not None else 1.0,
                    "reliability": float(rel.get(sid, np.nan)),
                    "rationale": text,
                }
            )
        s.recommendations = pd.DataFrame(recs)

        summ = s.inventory_summary.set_index("policy")
        fallback = (
            f"Agentic policy fill rate {summ.loc['agentic', 'fill_rate']:.1%} vs {summ.loc[classical_name, 'fill_rate']:.1%} "
            f"for the classical policy; stockout rate {summ.loc['agentic', 'stockout_rate']:.1%} vs "
            f"{summ.loc[classical_name, 'stockout_rate']:.1%}."
        )
        self.decide(s.llm.explain(fallback, fallback + " Summarise the trade-off with average stock held."), classical=classical_name)
        s.send(self.name, "Orchestrator", "Simulation and order recommendations complete.")
