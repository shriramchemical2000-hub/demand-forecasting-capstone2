"""Robustness and sensitivity experiments.

* ``run_seed_suite``   - repeat the whole pipeline on several independent synthetic worlds, so no result
                         depends on one random seed.
* ``run_sensitivity``  - re-plan on one trained pipeline while varying the service level, lead time and
                         cost parameters (the Planner is cheap to re-run; forecasts do not change).
"""
import copy
import json
from pathlib import Path

import yaml

import numpy as np
import pandas as pd

from .agents.planner import PlannerAgent
from .agents.state import PipelineState
from .orchestrator import run_pipeline
from .tracing import TraceLogger


def replan(state: PipelineState, inventory_overrides: dict) -> pd.DataFrame:
    """Re-run only the Planner with modified inventory settings; returns the policy summary."""
    cfg = copy.deepcopy(state.cfg)
    for k, v in inventory_overrides.items():
        cfg["inventory"][k] = v
    clone = copy.copy(state)
    clone.cfg = cfg
    clone.tracer = TraceLogger()  # silent
    PlannerAgent(clone).run()
    return clone.inventory_summary


def efficiency_frontier(service_df: pd.DataFrame, targets=None) -> pd.DataFrame:
    """Average stock each policy needs to reach a target fill rate (interpolated along the service-level sweep).

    This is the fair comparison: a policy that simply holds more stock should not look better.
    Targets default to the 25th / 50th / 75th percentile of all fill rates in the sweep.
    NaN means the policy never reaches that fill rate inside the sweep (service level <= 99.5%); if the target is
    below everything the policy achieved, the stock at its lowest service level is reported (an upper bound).
    """
    if targets is None:
        targets = [round(float(q), 3) for q in service_df["fill_rate"].quantile([0.25, 0.5, 0.75])]
    rows = []
    for target in targets:
        for pol, g in service_df.groupby("policy"):
            g = g.sort_values("fill_rate")
            ok = target <= g["fill_rate"].max()  # below the swept range -> value at the lowest service level (an upper bound)
            rows.append({"target_fill_rate": target, "policy": pol,
                         "avg_stock_needed": float(np.interp(target, g["fill_rate"], g["avg_on_hand_units"])) if ok else np.nan})
    return pd.DataFrame(rows)


def run_seed_suite(cfg: dict, seeds, out_dir, verbose=True) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c["project"]["seed"] = int(seed)
        if verbose:
            print(f"-- seed {seed}")
        st = run_pipeline(c, out / f"seed_{seed}", verbose=False)
        summ = st.metrics["summary"]
        fm = pd.DataFrame(summ["forecast_metrics"])
        im = pd.DataFrame(summ["interval_metrics"]).set_index("segment")
        for _, r in st.inventory_summary.iterrows():
            rows.append({"seed": seed, "kind": "inventory", "name": r["policy"], "segment": "all",
                         **{k: r[k] for k in ["fill_rate", "stockout_rate", "avg_on_hand_units", "total_cost"]}})
        for _, r in fm.iterrows():
            rows.append({"seed": seed, "kind": "forecast", "name": r["model"], "segment": r["segment"], "wape": r["wape"]})
        for seg, r in im.iterrows():
            rows.append({"seed": seed, "kind": "interval", "name": "agentic_intervals", "segment": seg, "coverage": r["coverage"]})
        crit = summ["critic"]
        rows.append({"seed": seed, "kind": "critic", "name": "critic", "segment": "all",
                     "precision": crit.get("precision"), "recall": crit.get("recall")})
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "seeds_raw.csv", index=False)
    metrics = [c for c in raw.columns if c not in ("seed", "kind", "name", "segment")]
    agg = raw.groupby(["kind", "name", "segment"])[metrics].agg(["mean", "std"]).dropna(axis=1, how="all")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "seeds_summary.csv", index=False)
    return agg


def run_sensitivity(state: PipelineState, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = state.cfg["inventory"]
    H = state.cfg["forecast"]["horizon"]
    res = {}

    def tag(df, **kw):
        return df.assign(**kw)

    service = pd.concat(
        [tag(replan(state, {"service_level": sl}), service_level=sl) for sl in (0.80, 0.90, 0.95, 0.975, 0.99, 0.995)]
    )
    res["service_level"] = service
    res["frontier"] = efficiency_frontier(service)

    lead = pd.concat(
        [tag(replan(state, {"lead_time_days": L}), lead_time=L)
         for L in range(0, H) if L + base["review_period_days"] <= H]
    )
    res["lead_time"] = lead

    grid = []
    for hr in (0.10, 0.25, 0.50):
        for sc in (0.2, 0.5, 1.0):
            grid.append(tag(replan(state, {"holding_rate_annual": hr, "stockout_cost_frac_of_price": sc}),
                            holding_rate_annual=hr, stockout_cost_frac=sc))
    res["cost_params"] = pd.concat(grid)

    for name, df in res.items():
        df.to_csv(out / f"sensitivity_{name}.csv", index=False)
    return res


def load_state_from_outputs(out_dir) -> PipelineState:
    """Rebuild just enough pipeline state from a results folder to re-run the Planner (used by the dashboard)."""
    from .llm import LLMHelper

    out = Path(out_dir)
    cfg = yaml.safe_load((out / "config_used.yaml").read_text())
    f = pd.read_csv(out / "forecasts.csv", parse_dates=["date"])
    summary = json.loads((out / "metrics.json").read_text())
    st = PipelineState(cfg=cfg, tracer=TraceLogger(), llm=LLMHelper({"llm": {"enabled": False}}))
    flag_cols = [c for c in ["event_flag", "weather_flag", "drift_flag", "price_flag"] if c in f.columns]
    st.risk = f[["date", "series_id", *flag_cols, "risk_multiplier"]].copy()
    st.forecasts = f.drop(columns=flag_cols + ["risk_multiplier"])
    if "ets" in f.columns and f["ets"].notna().any():
        st.baselines["ets"] = f[["series_id", "date", "ets"]].dropna().rename(columns={"ets": "y_hat"})
        st.baseline_sigma["ets"] = json.loads((out / "baseline_sigma.json").read_text()).get("ets", {})
    rel = out / "reliability.csv"
    st.reliability = pd.read_csv(rel) if rel.exists() else None
    prices = pd.read_csv(out / "prices.csv")
    st.price = prices.set_index("series_id")["price"]
    st.panel = f[["date", "series_id"]].merge(prices.rename(columns={"price": "sell_price"}), on="series_id")
    st.psum_inflation = float(summary.get("psum_inflation", 1.0))
    return st


SERVICE_LEVELS = (0.80, 0.90, 0.95, 0.975, 0.99, 0.995)


def stock_vs_reference(service_df: pd.DataFrame, reference: str = "gbm_static", ref_levels=(0.90, 0.95, 0.975)) -> pd.DataFrame:
    """For every policy: stock needed to reach the fill rates the reference policy reaches at ``ref_levels``,
    as a ratio to the reference's own stock (1.00 = same, below 1.00 = less stock for the same service).
    Policies that never reach a target get NaN for it.
    """
    ref = service_df[service_df["policy"] == reference].set_index("service_level")
    targets = [(ref.loc[lv, "fill_rate"], ref.loc[lv, "avg_on_hand_units"]) for lv in ref_levels if lv in ref.index]
    rows = []
    for pol, g in service_df.groupby("policy"):
        g = g.sort_values("fill_rate")
        ratios = []
        for fill, ref_stock in targets:
            ok = g["fill_rate"].min() - 1e-9 <= fill <= g["fill_rate"].max() + 1e-9
            ratios.append(float(np.interp(fill, g["fill_rate"], g["avg_on_hand_units"])) / ref_stock if ok else np.nan)
        rows.append({"policy": pol, **{f"ratio_{i}": r for i, r in enumerate(ratios)},
                     "mean_ratio": float(np.nanmean(ratios)) if np.isfinite(ratios).any() else np.nan})
    return pd.DataFrame(rows)


def run_frontier_worlds(cfg: dict, seeds, shocks: bool, learn_weights: bool, out_dir=None, verbose=True) -> pd.DataFrame:
    """Same-service stock comparison across independent synthetic worlds (calm or with shocks)."""
    out = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c["project"]["seed"] = int(seed)
        c["data"]["synthetic"]["shocks"] = bool(shocks)
        c["critic"]["learn_weights"] = bool(learn_weights)
        st = run_pipeline(c, Path(out_dir or "/tmp/frontier_runs") / f"s{seed}", verbose=False)
        sweep = pd.concat([replan(st, {"service_level": sl}).assign(service_level=sl) for sl in SERVICE_LEVELS])
        r = stock_vs_reference(sweep).assign(seed=seed, shocks=shocks, learn_weights=learn_weights)
        out.append(r)
        if verbose:
            print(f"seed {seed} done")
    df = pd.concat(out, ignore_index=True)
    return df
