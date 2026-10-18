#!/usr/bin/env python
"""Generate a first draft of the final report (markdown) from the results folders.

    python scripts/build_report.py --results outputs/synthetic --experiments outputs/experiments --out outputs/report_draft.md

Sections that need people (real-data discussion, qualitative review, conclusions) are left as clearly marked TODOs.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def table(df: pd.DataFrame, floatfmt="{:.3f}", na="n/a") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [na if (isinstance(v, float) and v != v) else (floatfmt.format(v) if isinstance(v, float) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/synthetic")
    ap.add_argument("--experiments", default="outputs/experiments")
    ap.add_argument("--out", default="outputs/report_draft.md")
    a = ap.parse_args()
    res, exp = Path(a.results), Path(a.experiments)
    m = json.loads((res / "metrics.json").read_text())
    cfg_src = (res / "config_used.yaml").read_text().split("source:")[1].split()[0]

    fm = pd.read_csv(res / "forecast_metrics.csv").pivot(index="model", columns="segment", values="wape").reset_index()
    im = pd.read_csv(res / "interval_metrics.csv")
    inv = pd.read_csv(res / "inventory_summary.csv")[["policy", "fill_rate", "stockout_rate", "avg_on_hand_units", "total_cost"]]
    crit = m["critic"]
    fcast = m.get("forecaster", {})

    parts = [
        "# Agentic Demand Forecasting & Inventory Optimization - Final Report (DRAFT)",
        f"_Auto-generated from `{res}` (data source: **{cfg_src}**). Replace every TODO before submitting._",
        "## 1. Problem and objectives\nTODO: reuse the abstract from the first draft; state the four objectives of the brief.",
        "## 2. System design\nFour collaborating agents (Forecaster, Uncertainty, Critic, Planner) coordinated by an orchestrator; see `docs/architecture.md` "
        "and the diagram in the README. All reasoning is logged to `traces.jsonl`.",
        "## 3. Data\nTODO (Data Engineering Lead): sources, pilot subset, cleaning, data-quality findings, weather source.",
        "## 4. Quantitative evaluation",
        "### 4.1 Forecast accuracy on the test window (WAPE, lower is better)\n" + table(fm),
        f"The Forecaster chose **{fcast.get('choice', 'n/a')}** on held-out validation data.",
        "### 4.2 Prediction intervals (90% target)\n" + table(im[["segment", "n", "coverage", "mean_width"]]),
        "### 4.3 Critic quality\n" + "\n".join(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in crit.items()),
        "### 4.4 Inventory simulation (same demand for every policy)\n" + table(inv)
        + f"\n\nStock is measured in: {m.get('unit_label', 'units')}.",
    ]
    f = exp / "sensitivity_frontier.csv"
    if f.exists():
        fr = pd.read_csv(f).pivot(index="target_fill_rate", columns="policy", values="avg_stock_needed").reset_index()
        fr["target_fill_rate"] = fr["target_fill_rate"].map(lambda v: f"{v:.1%}")
        parts.append("### 4.5 Fair comparison: average stock needed for a target fill rate (lower is better)\n"
                     + table(fr, "{:.1f}", na="not reached") + "\n\n'not reached' = the policy never hits that fill rate within the tested service levels (80-99.5%).")
    s = exp / "seeds" / "seeds_summary.csv"
    if s.exists():
        sd = pd.read_csv(s)
        sd = sd[sd["kind"] == "inventory"][["name", "fill_rate_mean", "fill_rate_std", "total_cost_mean", "total_cost_std"]]
        parts.append("### 4.6 Robustness over independent synthetic worlds (mean and std)\n" + table(sd))
    parts += [
        "Figures: `outputs/figures/` (run `scripts/make_report_figures.py`).",
        "## 5. Qualitative evaluation\nTODO (Critic and Evaluation Lead): ratings from `scripts/export_trace_review.py`, quoted traces, lessons. See `docs/trace_review.md`.",
        "## 6. Limitations\nTODO: copy and update the 'Known limitations' of the README with what the real-data runs showed.",
        "## 7. Conclusions and future work\nTODO.",
        "## 8. Team contributions\nTODO: one paragraph per member, matching the roles in the first draft.",
    ]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n\n".join(parts) + "\n")
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
