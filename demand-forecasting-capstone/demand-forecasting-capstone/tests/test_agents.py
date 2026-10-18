import json
from pathlib import Path

import pandas as pd

from demand_agents.agents.critic import CriticAgent
from demand_agents.agents.forecaster import ForecasterAgent
from demand_agents.agents.state import PipelineState
from demand_agents.agents.uncertainty import UncertaintyAgent
from demand_agents.data.loaders import load_dataset
from demand_agents.llm import LLMHelper
from demand_agents.orchestrator import run_pipeline
from demand_agents.tracing import TraceLogger, read_trace


def test_end_to_end_pipeline_writes_outputs_and_traces(small_cfg):
    state = run_pipeline(small_cfg, verbose=False)
    out = Path(small_cfg["output_dir"])
    for name in ["forecasts.csv", "forecast_metrics.csv", "interval_metrics.csv", "inventory_summary.csv",
                 "inventory_daily.csv", "recommendations.csv", "metrics.json", "traces.jsonl"]:
        assert (out / name).exists(), name

    traces = read_trace(out / "traces.jsonl")
    assert {"Forecaster", "Uncertainty", "Critic", "Planner"} <= set(traces["agent"])   # >= 3 agents
    assert {"thought", "observation", "decision", "message"} <= set(traces["kind"])

    inv = state.inventory_summary.set_index("policy")
    assert {"agentic", "gbm_static"} <= set(inv.index)
    assert ((inv["fill_rate"] > 0) & (inv["fill_rate"] <= 1)).all()
    assert (state.risk["risk_multiplier"] >= 1).all()
    assert json.loads((out / "metrics.json").read_text())["n_series"] == 10


def _state_after_forecaster(cfg):
    tracer = TraceLogger()
    state = PipelineState(cfg=cfg, tracer=tracer, llm=LLMHelper(cfg))
    state.panel = load_dataset(cfg)
    ForecasterAgent(state).run()
    return state


def test_critic_requests_wider_intervals_when_undercovering(small_cfg):
    state = _state_after_forecaster(small_cfg)
    unc, critic = UncertaintyAgent(state), CriticAgent(state)
    unc.run()
    unc.revise(0.3)                                   # sabotage: intervals far too narrow
    review = critic.review_calibration()
    assert review["approved"] is False and review["scale"] > state.interval_scale  # asks to widen
    assert any(e["kind"] == "revision_request" for e in state.tracer.events)
    unc.revise(review["scale"])
    assert critic.review_calibration()["coverage"] > review["coverage"]


def test_trace_logger_roundtrip(tmp_path):
    t = TraceLogger(tmp_path / "t.jsonl")
    import numpy as np
    t.log("A", "thought", "hello", {"x": np.float64(1.5), "arr": np.arange(3)})
    df = read_trace(tmp_path / "t.jsonl")
    assert df.loc[0, "data"] == {"x": 1.5, "arr": [0, 1, 2]}


def test_weather_sensitivity_is_learned_and_risk_widens_intervals(small_cfg):
    from demand_agents.agents.critic import CriticAgent
    from demand_agents.agents.uncertainty import UncertaintyAgent

    small_cfg["data"]["synthetic"].update(n_stores=2, n_items=10, n_days=700)
    state = _state_after_forecaster(small_cfg)
    unc, critic = UncertaintyAgent(state), CriticAgent(state)
    unc.run()
    unc.finalize()
    critic.assess_risk()
    unc.apply_risk()

    sens = state.weather_sensitivity.merge(state.panel[["series_id", "category"]].drop_duplicates(), on="series_id")
    assert sens.loc[sens["category"] == "beverages", "sensitive"].mean() >= 0.9     # temperature-driven items found
    assert sens.loc[sens["category"] == "staples", "sensitive"].mean() <= 0.2       # insensitive items left alone

    f = state.forecasts
    assert ((f["upper"] - f["y_hat"]) >= (f["upper_base"] - f["y_hat"]) - 1e-9).all()
    assert (f["upper"] > f["upper_base"]).any()                                      # some days were widened


def test_replan_changes_only_the_planner(small_cfg):
    from demand_agents.experiments import replan

    state = run_pipeline(small_cfg, verbose=False)
    before = state.inventory_summary.copy()
    hi = replan(state, {"service_level": 0.99}).set_index("policy")
    lo = replan(state, {"service_level": 0.80}).set_index("policy")
    assert (hi["avg_on_hand_units"] > lo["avg_on_hand_units"]).all()
    pd.testing.assert_frame_equal(before, state.inventory_summary)                   # original state untouched
