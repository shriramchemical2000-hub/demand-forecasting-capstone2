"""Shared state that flows through the agent pipeline (the 'blackboard')."""
from dataclasses import dataclass, field

import pandas as pd

from ..llm import LLMHelper
from ..tracing import TraceLogger


@dataclass
class PipelineState:
    cfg: dict
    tracer: TraceLogger
    llm: LLMHelper
    panel: pd.DataFrame | None = None        # canonical long data
    feats: pd.DataFrame | None = None        # panel + features + split column
    features: list = field(default_factory=list)
    splits: dict | None = None
    model: object = None
    forecasts: pd.DataFrame | None = None    # val + test rows: y_hat, lower, upper, sigma
    calibrator: object = None
    interval_scale: float = 1.0
    psum_inflation: float = 1.0              # sigma correction for multi-period demand (autocorrelated errors)
    reliability: pd.DataFrame | None = None  # per series validation WAPE / reliability
    risk: pd.DataFrame | None = None         # Critic output: flags + risk multiplier
    weather_sensitivity: pd.DataFrame | None = None
    baselines: dict = field(default_factory=dict)
    baseline_sigma: dict = field(default_factory=dict)
    price: pd.Series | None = None           # median price per series over the test window (cost base)
    inventory_summary: pd.DataFrame | None = None
    inventory_daily: pd.DataFrame | None = None
    recommendations: pd.DataFrame | None = None
    metrics: dict = field(default_factory=dict)

    def send(self, sender: str, receiver: str, text: str, **data):
        self.tracer.log(sender, "message", f"-> {receiver}: {text}", data)
