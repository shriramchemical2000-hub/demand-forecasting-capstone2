import json

import numpy as np
import pandas as pd
import pytest

from demand_agents.agents.critic import CriticAgent
from demand_agents.agents.uncertainty import UncertaintyAgent
from demand_agents.evaluation.report import psum_calibration
from demand_agents.experiments import load_state_from_outputs, replan
from demand_agents.inventory.policies import base_stock_components
from demand_agents.inventory.simulator import simulate_base_stock, summarize
from demand_agents.orchestrator import run_pipeline

from test_agents import _state_after_forecaster


def _demand(T=80, N=6, lam=10.0, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.poisson(lam, (T, N)).astype(float)
    m, s = base_stock_components(np.full((T, N), lam), np.full((T, N), np.sqrt(lam)), 1.645, 3)
    return d, m + s, m


def test_batches_and_minimum_order_quantity():
    d, S, _ = _demand()
    sim = simulate_base_stock(d, S, 2, batch_size=12, min_order_qty=24)
    q = sim.orders[sim.orders > 0]
    assert (q % 12 == 0).all() and (q >= 24).all()


def test_supplier_delays_hurt_service():
    d, S, _ = _demand(T=200, lam=10.0, seed=3)
    base = simulate_base_stock(d, S, 2)
    late = simulate_base_stock(d, S, 2, delay_prob=0.5, delay_max=3, rng=np.random.default_rng(1))
    assert late.lost.sum() > base.lost.sum()


def test_s_S_rule_places_fewer_orders_and_fixed_cost_is_counted():
    d, S, m = _demand()
    base = simulate_base_stock(d, S, 2)
    sS = simulate_base_stock(d, S, 2, reorder_point=S - 2 * m / 3)
    assert sS.n_orders.sum() < base.n_orders.sum()
    h, c = np.full(d.shape[1], 0.01), np.full(d.shape[1], 1.0)
    s1 = summarize(base, d, h, c, fixed_order_cost=0.0)
    s2 = summarize(base, d, h, c, fixed_order_cost=5.0)
    assert s2["order_cost"] == 5.0 * base.n_orders.sum() and s2["total_cost"] == s1["total_cost"] + s2["order_cost"]


def test_adaptive_interval_scaling_respects_information_delay(small_cfg):
    state = _state_after_forecaster(small_cfg)
    unc = UncertaintyAgent(state)
    unc.run()
    unc.finalize()
    unc.revise(0.3)                        # intervals far too narrow -> many misses -> multiplier must rise
    state.risk = state.forecasts[["date", "series_id"]].assign(
        event_flag=False, weather_flag=False, drift_flag=False, price_flag=False, risk_multiplier=1.0)
    unc.apply_risk()
    unc.apply_adaptive()
    f = state.forecasts
    H = small_cfg["forecast"]["horizon"]
    first = f[f["split"] == "val_check"].sort_values("date").groupby("series_id").head(H)
    assert (first["aci_mult"] == 1.0).all()               # nothing is known yet for the first H periods
    assert f.loc[f["split"] == "test", "aci_mult"].max() > 1.2


def test_price_shock_signal_raises_risk(small_cfg):
    state = _state_after_forecaster(small_cfg)
    unc, critic = UncertaintyAgent(state), CriticAgent(state)
    unc.run()
    unc.finalize()
    test_rows = state.feats.index[state.feats["split"] == "test"][:5]
    state.feats.loc[test_rows, "price_rel"] = 0.2           # an unprecedented 80% discount
    critic.assess_risk()
    hit = state.risk.merge(state.feats.loc[test_rows, ["date", "series_id"]], on=["date", "series_id"])
    assert hit["price_flag"].all() and (hit["risk_multiplier"] >= 1 + small_cfg["critic"]["risk_weights"]["price"]).all()
    assert state.risk["price_flag"].dtype == bool


def test_abc_service_levels_and_buffer(small_cfg):
    small_cfg["inventory"]["abc_service_levels"]["enabled"] = True
    small_cfg["inventory"]["lead_time_buffer"] = 1
    state = run_pipeline(small_cfg, verbose=False)
    inv = state.inventory_summary.set_index("policy")
    assert "agentic_abc" in inv.index and inv["fill_rate"].between(0, 1).all()


def test_forecaster_selects_among_candidates_with_ets_blend(small_cfg):
    small_cfg["forecast"]["selection"] = {"blend_with_ets": True, "min_relative_gain": 0.02}
    small_cfg["data"]["synthetic"].update(n_stores=1, n_items=4)
    state = run_pipeline(small_cfg, verbose=False)
    info = state.metrics["summary"]["forecaster"]
    assert info["choice"] in {"gbm", "blend", "seasonal_naive"} and 0.0 <= info["blend_weight"] <= 1.0
    assert "ets" in state.baselines


def test_replan_from_saved_outputs_reproduces_the_planner(small_cfg):
    small_cfg["forecast"]["selection"] = {"blend_with_ets": True}
    small_cfg["forecast"]["baselines"] = ["seasonal_naive", "ets"]
    small_cfg["data"]["synthetic"].update(n_stores=1, n_items=4)
    state = run_pipeline(small_cfg, verbose=False)
    st2 = load_state_from_outputs(small_cfg["output_dir"])
    again = replan(st2, {}).set_index("policy")
    saved = state.inventory_summary.set_index("policy")
    for col in ["fill_rate", "total_cost", "avg_on_hand_units"]:
        np.testing.assert_allclose(again[col], saved.loc[again.index, col], rtol=1e-5)


def test_psum_calibration_detects_autocorrelated_errors():
    rng = np.random.default_rng(0)
    n = 600
    e = np.convolve(rng.normal(0, 1, n + 2), np.ones(3) / np.sqrt(3), mode="valid")  # strongly autocorrelated errors
    t = pd.DataFrame({"series_id": "a", "date": pd.date_range("2022-01-01", periods=len(e)),
                      "sales": 10 + e, "y_hat": 10.0, "sigma": 1.0})
    out = psum_calibration(t, 3, 1.0)
    assert out["z_std_raw"] > 1.3                          # naive sqrt(sum sigma^2) is too small


def test_lightgbm_model_option(small_cfg):
    pytest.importorskip("lightgbm")
    small_cfg["forecast"]["model"]["type"] = "lightgbm"
    small_cfg["data"]["synthetic"].update(n_stores=1, n_items=3)
    state = run_pipeline(small_cfg, verbose=False)
    assert state.inventory_summary["fill_rate"].between(0, 1).all()


def test_classical_baseline_survives_nan_fits_and_missing_history(monkeypatch):
    import warnings

    from demand_agents.data.synthetic import generate_synthetic
    from demand_agents.models import baselines

    df = generate_synthetic(n_stores=1, n_items=3, n_days=500, seed=4)
    df.loc[(df["series_id"] == df["series_id"].iloc[0]) & (df["date"] < df["date"].min() + pd.Timedelta(days=250)), "sales"] = np.nan

    def nan_fit(y, h):
        return np.full(h, np.nan), float("nan")          # what a failing statsmodels fit can return

    monkeypatch.setattr(baselines, "_ets_fit_factory", lambda period: nan_fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, sigma = baselines.rolling_classical_forecast(df.dropna(subset=["sales"]).assign(sales=lambda d: d["sales"]),
                                                          df["date"].max() - pd.Timedelta(days=28), 7, "ets")
    assert np.isfinite(out["y_hat"]).all() and all(np.isfinite(v) and v > 0 for v in sigma.values())


def test_planner_survives_unusable_classical_inputs(small_cfg):
    small_cfg["forecast"]["selection"] = {"blend_with_ets": True}
    small_cfg["forecast"]["baselines"] = ["seasonal_naive", "ets"]
    small_cfg["data"]["synthetic"].update(n_stores=1, n_items=4)
    state = run_pipeline(small_cfg, verbose=False)
    first = next(iter(state.baseline_sigma["ets"]))
    state.baseline_sigma["ets"][first] = float("nan")             # poison one series
    state.baselines["ets"].loc[state.baselines["ets"]["series_id"] == first, "y_hat"] = float("nan")
    out = replan(state, {}).set_index("policy")
    assert np.isfinite(out.loc["classical_ets", ["fill_rate", "total_cost", "avg_on_hand_units"]]).all()


def test_critic_learns_weights_from_validation_errors(small_cfg):
    from demand_agents.agents.state import PipelineState
    from demand_agents.llm import LLMHelper
    from demand_agents.tracing import TraceLogger

    state = PipelineState(cfg=small_cfg, tracer=TraceLogger(), llm=LLMHelper(small_cfg))
    critic = CriticAgent(state)
    n = 4000
    rng = np.random.default_rng(0)
    f = pd.DataFrame({
        "split": "val_cal",
        "nz": rng.gamma(4, 0.25, n),                         # normalised errors with mean ~1
        "ref_nz": 1.0,
        "event_flag": rng.random(n) < 0.2,                   # fires often but does NOT predict larger errors
        "weather_flag": False,                               # never fires on validation
        "price_flag": False,
        "drift_flag": False,
        "drift_excess": 0.0,
    })
    w = critic._learn_weights(f, small_cfg["critic"])
    prior = small_cfg["critic"]["risk_weights"]
    assert w["event"] < 0.3 * prior["event"]                 # pushed towards zero by the evidence
    assert w["weather"] == prior["weather"] and w["price"] == prior["price"]   # no evidence -> keep the prior
    assert w["drift"] == prior["drift"]

    f["nz"] = np.where(f["event_flag"], f["nz"] * 1.8, f["nz"])   # now events really do raise the error
    w2 = critic._learn_weights(f, small_cfg["critic"])
    assert w2["event"] > 0.6                                  # learns a large weight when the signal is informative


def test_nowcast_is_optional_and_off_by_default(small_cfg):
    assert small_cfg["inventory"]["nowcast"]["enabled"] is False
    off = run_pipeline(small_cfg, verbose=False).inventory_summary["policy"].tolist()
    assert "agentic" in off and "agentic_risk" not in off          # without the nowcast the two would be identical
    small_cfg["inventory"]["nowcast"]["enabled"] = True
    on = run_pipeline(small_cfg, verbose=False).inventory_summary["policy"].tolist()
    assert {"agentic", "agentic_risk"} <= set(on)
