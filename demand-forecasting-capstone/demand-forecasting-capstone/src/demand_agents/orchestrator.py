"""Orchestrator: runs the agents in order and handles the Critic <-> Uncertainty revision loop."""
import json
from pathlib import Path

from .agents.critic import CriticAgent
from .agents.forecaster import ForecasterAgent
from .agents.planner import PlannerAgent
from .agents.state import PipelineState
from .agents.uncertainty import UncertaintyAgent
from .data.loaders import load_dataset
from .evaluation.report import evaluate_and_save
from .llm import LLMHelper
from .models.baselines import rolling_classical_forecast
from .tracing import TraceLogger


def run_pipeline(cfg: dict, output_dir=None, verbose: bool = True) -> PipelineState:
    out = Path(output_dir or cfg.get("output_dir", "outputs/latest"))
    tracer = TraceLogger(out / "traces.jsonl")
    state = PipelineState(cfg=cfg, tracer=tracer, llm=LLMHelper(cfg))

    def say(msg):
        if verbose:
            print(msg)

    tracer.log("Orchestrator", "thought", "Plan: Forecaster -> Uncertainty <-> Critic (revision loop) -> Critic risk -> Planner.")

    say("[1/6] Loading data ...")
    state.panel = load_dataset(cfg)
    tracer.log(
        "Orchestrator", "action",
        f"Loaded {len(state.panel):,} rows, {state.panel['series_id'].nunique()} series, source={cfg['data']['source']}.",
    )

    from .data.audit import audit_panel

    audit = audit_panel(state.panel, cfg["data"].get("frequency", "D"), cfg["data"]["source"], cfg["data"].get("season_period", 7))
    tracer.log("Orchestrator", "observation",
               f"Data audit: {audit['overview']['series']} series, zero share {audit['demand']['zero_share']:.1%}; "
               f"findings: {', '.join(i['check'] for i in audit['issues']) or 'none'}.", {"issues": audit["issues"]})
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))

    say("[2/6] Forecaster ...")
    ForecasterAgent(state).run()

    say("[3/6] Uncertainty <-> Critic ...")
    uncertainty, critic = UncertaintyAgent(state), CriticAgent(state)
    uncertainty.run()
    for rnd in range(1, cfg["critic"]["max_revision_rounds"] + 1):
        review = critic.review_calibration()
        if review["approved"]:
            break
        uncertainty.revise(review["scale"])
    else:
        review = critic.review_calibration()
        if not review["approved"]:
            tracer.log("Critic", "decision", "Revision limit reached; continuing with a calibration warning.")
    uncertainty.finalize()

    say("[4/6] Critic risk assessment ...")
    critic.assess_risk()
    uncertainty.apply_risk()
    uncertainty.apply_adaptive()

    say("[5/6] Classical baselines ...")
    fc = cfg["forecast"]
    for name in fc["baselines"]:
        if name == "seasonal_naive" or name in state.baselines:
            continue
        tracer.log("Orchestrator", "action", f"Running {name} baseline (rolling origin).")
        df, sigma = rolling_classical_forecast(
            state.panel, state.splits["test_start"], fc["horizon"], method=name,
            arima_order=fc.get("arima_order", (7, 0, 0)), window=fc.get("classical_window", 365),
            season_period=cfg["data"].get("season_period", 7),
        )
        state.baselines[name] = df
        state.baseline_sigma[name] = sigma

    say("[6/6] Planner + evaluation ...")
    PlannerAgent(state).run()
    summary = evaluate_and_save(state, out)
    tracer.log("Orchestrator", "decision", f"Run complete. Outputs written to {out}.", {"output_dir": str(out)})
    say(f"Done. Outputs in {out}")
    return state
