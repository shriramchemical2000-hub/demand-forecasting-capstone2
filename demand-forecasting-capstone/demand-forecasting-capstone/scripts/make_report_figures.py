#!/usr/bin/env python
"""Create report-ready PNG figures from a results folder.

    python scripts/make_report_figures.py --results outputs/synthetic --experiments outputs/experiments --out outputs/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/synthetic")
    ap.add_argument("--experiments", default="outputs/experiments")
    ap.add_argument("--out", default="outputs/figures")
    a = ap.parse_args()
    res, exp, out = Path(a.results), Path(a.experiments), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fc = pd.read_csv(res / "forecasts.csv", parse_dates=["date"])
    summ = json.loads((res / "metrics.json").read_text())
    unit = summ.get("unit_label", "units")
    test = fc[fc["split"] == "test"]
    written = []

    # 1. forecast with interval and Critic flags for the series with the largest shock-period error
    err = (test["sales"] - test["y_hat"]).abs().groupby(test["series_id"]).sum()
    sid = err.idxmax()
    s = fc[(fc["series_id"] == sid) & fc["split"].isin(["val_check", "test"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(s["date"], s["lower"], s["upper"], alpha=0.2, label="90% interval")
    ax.plot(s["date"], s["sales"], "k.-", lw=0.8, label="actual")
    ax.plot(s["date"], s["y_hat"], label="forecast")
    if "ets" in s.columns:
        ax.plot(s["date"], s["ets"], ":", label="ETS baseline")
    fl = s[(s["risk_multiplier"] > 1.2)]
    ax.scatter(fl["date"], fl["sales"], marker="D", facecolors="none", edgecolors="crimson", label="Critic flag")
    ax.axvline(pd.Timestamp(summ["splits"]["test_start"]), ls="--", c="gray")
    ax.set_title(f"Forecast, interval and Critic flags: {sid}")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig1_forecast_example.png"); plt.close(fig); written.append("fig1_forecast_example.png")

    # 2. forecast error by segment
    fm = pd.read_csv(res / "forecast_metrics.csv").pivot(index="model", columns="segment", values="wape")
    ax = fm.plot.bar(figsize=(7, 4), rot=0)
    ax.set_ylabel("WAPE (lower is better)"); ax.set_title("Forecast error by segment")
    plt.tight_layout(); plt.savefig(out / "fig2_wape_by_segment.png"); plt.close(); written.append("fig2_wape_by_segment.png")

    # 3. interval coverage by segment
    im = pd.read_csv(res / "interval_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(im["segment"], im["coverage"]); ax.axhline(0.9, ls="--", c="k")
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("coverage"); ax.set_title("Prediction-interval coverage (90% target)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(out / "fig3_interval_coverage.png"); plt.close(fig); written.append("fig3_interval_coverage.png")

    # 4. policy ladder
    inv = pd.read_csv(res / "inventory_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, title in zip(axes, ["fill_rate", "avg_on_hand_units", "total_cost"], ["Fill rate", f"Average stock ({unit})", "Total cost"]):
        ax.bar(inv["policy"], inv[col]); ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(max(0, inv["fill_rate"].min() - 0.03), 1.0)
    fig.suptitle("Policy ladder (same test window, same demand)"); fig.tight_layout()
    fig.savefig(out / "fig4_policy_ladder.png"); plt.close(fig); written.append("fig4_policy_ladder.png")

    # 5. reliability diagram
    if (res / "calibration_curve.csv").exists():
        cc = pd.read_csv(res / "calibration_curve.csv")
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot([0.5, 0.95], [0.5, 0.95], "--", c="gray")
        for seg, g in cc.groupby("segment"):
            ax.plot(g["nominal"], g["empirical"], "o-", label=seg)
        ax.set_xlabel("nominal"); ax.set_ylabel("empirical"); ax.set_title("Reliability diagram"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig5_reliability.png"); plt.close(fig); written.append("fig5_reliability.png")

    # 6. efficiency frontier and sensitivity (if experiments exist)
    f = exp / "sensitivity_service_level.csv"
    if f.exists():
        df = pd.read_csv(f)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for pol, g in df.groupby("policy"):
            g = g.sort_values("service_level")
            axes[0].plot(g["avg_on_hand_units"], g["fill_rate"], "o-", label=pol)
            axes[1].plot(g["service_level"], g["total_cost"], "o-", label=pol)
        axes[0].set_xlabel(f"average stock ({unit})"); axes[0].set_ylabel("fill rate"); axes[0].set_title("Efficiency frontier (up and left is better)")
        axes[1].set_xlabel("service-level setting"); axes[1].set_ylabel("total cost"); axes[1].set_title("Total cost vs service level")
        axes[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "fig6_frontier_and_cost.png"); plt.close(fig); written.append("fig6_frontier_and_cost.png")

    print("Wrote:", ", ".join(written), "->", out)


if __name__ == "__main__":
    main()
