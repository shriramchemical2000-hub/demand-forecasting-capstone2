#!/usr/bin/env python
"""Run the full agentic pipeline.

Examples
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --source m5 --output-dir outputs/m5_pilot
    python scripts/run_pipeline.py --config configs/walmart.yaml --source walmart_synthetic   # weekly mode demo
    python scripts/run_pipeline.py --baselines seasonal_naive ets arima
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_agents.config import load_config  # noqa: E402
from demand_agents.orchestrator import run_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", choices=["synthetic", "m5", "walmart", "walmart_synthetic"], default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--baselines", nargs="+", default=None)
    ap.add_argument("--service-level", type=float, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.source:
        overrides["data.source"] = args.source
    if args.baselines:
        overrides["forecast.baselines"] = args.baselines
    if args.service_level:
        overrides["inventory.service_level"] = args.service_level

    cfg = load_config(args.config, overrides)
    state = run_pipeline(cfg, args.output_dir)

    print("\n=== Forecast accuracy on the test window (WAPE, lower is better) ===")
    import pandas as pd
    fm = pd.DataFrame(state.metrics["summary"]["forecast_metrics"])
    print(fm.pivot(index="model", columns="segment", values="wape").round(3).to_string())
    print("\n=== Inventory simulation ===")
    inv = state.inventory_summary.set_index("policy")
    print(inv[["fill_rate", "stockout_rate", "avg_on_hand_units", "holding_cost", "stockout_cost", "total_cost"]].round(3).to_string())


if __name__ == "__main__":
    main()
