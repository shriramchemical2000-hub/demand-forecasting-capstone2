"""Command line interface:  demand-agents <command>   (or  python -m demand_agents <command>)

  download   fetch the raw data into data/raw/          (Kaggle token needed for M5 / Walmart)
  build      raw -> clean -> audit -> merge -> data/processed/ + reports/tables/
  eda        data audit + exploratory analysis -> reports/data_audit_<name>.md and reports/figures/
  run        the agent pipeline (forecast, uncertainty, critic, planner) -> outputs/
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .config import NAMED, load_config, load_named


def _cfg(args):
    overrides = {kv.split("=", 1)[0]: yaml.safe_load(kv.split("=", 1)[1]) for kv in getattr(args, "set", []) or []}
    if args.config in NAMED:
        return load_named(args.config, overrides)
    return load_config(args.config, overrides or None)


def cmd_download(args):
    from .data.acquire import download

    for p in download(args.dataset, args.root):
        print("ready:", p)


def cmd_build(args):
    from .data.integrate import build_dataset

    cfg = _cfg(args)
    s = build_dataset(cfg, reports_dir=args.reports)
    print(f"Built '{s['name']}': {s['rows_final']:,} rows -> {s.get('processed_path')}")
    for step in s["cleaning_log"]:
        print(f"  - {step['step']}: {step['rows_affected']:,} {step['note']}")
    for i in s["audit_final"]["issues"]:
        print(f"  [{i['severity']}] {i['check']}: {i['detail']}")


def cmd_eda(args):
    from .data.audit import audit_panel
    from .data.loaders import load_dataset
    from .eda.plots import make_figures
    from .eda.report import write_report
    from .eda.tables import run_eda

    cfg = _cfg(args)
    name = cfg["data"]["source"]
    reports = Path(args.reports)
    df = load_dataset(cfg)
    audit = audit_panel(df, cfg["data"].get("frequency", "D"), name, cfg["data"].get("season_period", 7))
    tables = run_eda(df)
    figs = make_figures(tables, df, reports / "figures", name)
    log_path = reports / "tables" / f"cleaning_log_{name}.csv"
    cleaning = pd.read_csv(log_path).fillna("").to_dict("records") if log_path.exists() else None
    js = reports / "tables" / f"data_audit_{name}.json"
    weather_audit = json.loads(js.read_text()).get("weather_audit") if js.exists() else None
    out = reports / f"data_audit_{name}.md"
    write_report(out, audit, tables, cleaning, weather_audit, figs)
    reports.joinpath("tables").mkdir(parents=True, exist_ok=True)
    tables["abc_summary"].to_csv(reports / "tables" / f"abc_summary_{name}.csv", index=False)
    tables["classes"].to_csv(reports / "tables" / f"demand_classes_{name}.csv", index=False)
    print("Wrote", out, "and", len(figs), "figures to", reports / "figures")


def cmd_run(args):
    from .orchestrator import run_pipeline

    cfg = _cfg(args)
    state = run_pipeline(cfg, args.output_dir)
    print(state.inventory_summary[["policy", "fill_rate", "stockout_rate", "avg_on_hand_units", "total_cost"]].round(3).to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="demand-agents", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--dataset", choices=["m5", "walmart", "noaa", "all"], default="all")
    d.add_argument("--root", default="data/raw")
    d.set_defaults(fn=cmd_download)

    for name, fn in (("build", cmd_build), ("eda", cmd_eda), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("--config", default=None, help="a dataset name (" + ", ".join(NAMED) + ") or a YAML file (default: configs/base.yaml)")
        s.add_argument("--set", action="append", default=[], help="override, e.g. --set data.m5.n_items=20 (repeatable)")
        s.add_argument("--reports", default="reports")
        if name == "run":
            s.add_argument("--output-dir", default=None)
        s.set_defaults(fn=fn)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
