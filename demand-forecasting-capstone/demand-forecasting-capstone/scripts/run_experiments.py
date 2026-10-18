#!/usr/bin/env python
"""Robustness + sensitivity experiments.

    python scripts/run_experiments.py seeds --n 5          # 5 independent synthetic worlds
    python scripts/run_experiments.py sensitivity          # service level / lead time / cost sweeps
    python scripts/run_experiments.py all
Results go to outputs/experiments/ and are shown in the dashboard's "Experiments" tab.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from demand_agents.config import load_config  # noqa: E402
from demand_agents.experiments import run_seed_suite, run_sensitivity  # noqa: E402
from demand_agents.orchestrator import run_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=["seeds", "sensitivity", "all", "frontier"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=5, help="number of seeds")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--source", default=None, help="override data.source, e.g. walmart_synthetic")
    ap.add_argument("--shocks", choices=["true", "false"], default="true", help="frontier suite: shocks in the test window?")
    ap.add_argument("--learn", choices=["true", "false"], default="true", help="frontier suite: learn the Critic weights?")
    ap.add_argument("--first-seed", type=int, default=301)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. inventory.nowcast.z_threshold=3.0 (repeatable)")
    args = ap.parse_args()
    import yaml
    overrides = {"data.source": args.source} if args.source else {}
    overrides.update({kv.split("=", 1)[0]: yaml.safe_load(kv.split("=", 1)[1]) for kv in args.set})
    cfg = load_config(args.config, overrides or None)
    pd.set_option("display.width", 200)

    if args.suite == "frontier":
        from demand_agents.experiments import run_frontier_worlds
        df = run_frontier_worlds(cfg, range(args.first_seed, args.first_seed + args.n), args.shocks == "true", args.learn == "true")
        print(f"\n=== Stock needed for the same service, relative to gbm_static (shocks={args.shocks}, learned weights={args.learn}) ===")
        print(df.groupby("policy")["mean_ratio"].agg(["mean", "std"]).round(3).to_string())
        Path(args.out).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(args.out) / f"frontier_shocks-{args.shocks}_learn-{args.learn}.csv", index=False)
        return

    if args.suite in ("seeds", "all"):
        agg = run_seed_suite(cfg, range(101, 101 + args.n), Path(args.out) / "seeds")
        cols = [c for c in agg.columns if c.endswith(("_mean", "_std")) or c in ("kind", "name", "segment")]
        print("\n=== Mean and std over seeds ===")
        print(agg[cols].round(3).to_string(index=False))

    if args.suite in ("sensitivity", "all"):
        state = run_pipeline(cfg, Path(args.out) / "base_run", verbose=False)
        res = run_sensitivity(state, Path(args.out))
        for name, df in res.items():
            if name == "frontier":
                print("\n=== Average stock needed to reach a target fill rate (lower is better; NaN = not reachable) ===")
                print(df.pivot(index="target_fill_rate", columns="policy", values="avg_stock_needed").round(1).to_string())
                continue
            print(f"\n=== Sensitivity: {name} (total cost) ===")
            keys = [c for c in df.columns if c not in ("policy", "fill_rate", "stockout_rate", "lost_units",
                    "avg_on_hand_units", "holding_cost", "stockout_cost", "total_cost")]
            print(df.pivot_table(index=keys, columns="policy", values="total_cost").round(0).to_string())


if __name__ == "__main__":
    main()
