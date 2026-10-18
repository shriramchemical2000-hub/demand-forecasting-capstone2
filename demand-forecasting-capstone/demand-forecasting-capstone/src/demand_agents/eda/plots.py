"""EDA figures (matplotlib, no display needed)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def make_figures(t: dict, df: pd.DataFrame, out_dir, name: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    def save(fig, key):
        path = out / f"{name}_{key}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        files[key] = str(path)

    # 1 total demand over time
    agg = t["aggregate"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(agg.index, agg.values, lw=0.6, alpha=0.6, label="total")
    ax.plot(agg.index, agg.rolling(28 if len(agg) > 200 else 4, min_periods=1).mean(), lw=1.8, label="rolling mean")
    ax.set_title("Total demand over time"); ax.legend()
    save(fig, "total_demand")

    # 2 seasonality
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    wd = t["weekday"]
    axes[0].bar(wd["weekday"], wd["rel"]); axes[0].axhline(1, c="k", lw=0.8); axes[0].set_title("Weekday profile (1.0 = series average)")
    mo = t["month"]
    axes[1].bar(mo["month"], mo["rel"]); axes[1].axhline(1, c="k", lw=0.8); axes[1].set_title("Month profile")
    save(fig, "seasonality")

    # 3 events
    ev = t["events"]
    if ev.get("available"):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.bar(["normal days", "event days"], [ev["non_event_mean_rel"], ev["event_mean_rel"]], color=["#7f9fbf", "#c0504d"])
        ax.set_title(f"Event effect ({ev['uplift']:+.0%})")
        save(fig, "events")

    # 4 intermittency
    c = t["classes"].replace([np.inf], np.nan).dropna(subset=["adi", "cv2"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(c["adi"], c["cv2"], s=14, alpha=0.7)
    ax.axvline(1.32, c="gray", ls="--"); ax.axhline(0.49, c="gray", ls="--")
    ax.set_xlabel("average demand interval (ADI)"); ax.set_ylabel("CV$^2$ of non-zero demand")
    ax.set_title("Demand types: smooth / erratic / intermittent / lumpy")
    save(fig, "intermittency")

    # 5 price response
    d = t["relative"]
    if "sell_price" in d.columns and d["sell_price"].nunique() > 1:
        x = d.dropna(subset=["sell_price", "rel"]).copy()
        x["pr"] = x["sell_price"] / x.groupby("series_id")["sell_price"].transform("mean")
        x["bin"] = pd.cut(x["pr"], [0, 0.7, 0.8, 0.9, 0.97, 1.03, 1.1, 1.25, 10])
        g = x.groupby("bin", observed=True)["rel"].agg(["mean", "count"]).reset_index()
        g = g[g["count"] > 20]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.bar(g["bin"].astype(str), g["mean"]); ax.axhline(1, c="k", lw=0.8)
        ax.set_title("Relative sales by price level (price / series mean price)"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "price_response")

    # 6 weather
    if t["weather"].get("available"):
        x = d.dropna(subset=["temp", "rel"]).copy()
        x["month"] = pd.to_datetime(x["date"]).dt.month
        x["anom"] = x["temp"] - x.groupby(["store_id", "month"])["temp"].transform("mean")
        x["rel_m"] = x["sales"] / x.groupby(["series_id", "month"])["sales"].transform("mean").replace(0, np.nan)
        x["bin"] = pd.qcut(x["anom"], 10, duplicates="drop")
        g = x.groupby("bin", observed=True)["rel_m"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(range(len(g)), g["rel_m"], "o-"); ax.axhline(1, c="k", lw=0.8)
        ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{i.mid:+.0f}" for i in g["bin"]])
        ax.set_xlabel("temperature anomaly (deg C, deciles)"); ax.set_title("Sales vs temperature anomaly (within month)")
        save(fig, "weather_response")

    # 7 volume concentration
    abc = t["abc"]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(np.arange(1, len(abc) + 1) / len(abc), abc["cum_share"]); ax.axhline(0.8, c="gray", ls="--")
    ax.set_xlabel("share of series (largest first)"); ax.set_ylabel("share of total volume"); ax.set_title("Volume concentration (ABC)")
    save(fig, "volume_concentration")

    # 8 zeros
    z = (df["sales"] == 0).groupby(df["series_id"]).mean()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.hist(z.values, bins=20); ax.set_xlabel("share of periods with zero sales"); ax.set_ylabel("series"); ax.set_title("Zero sales per series")
    save(fig, "zero_share")

    # 9 autocorrelation
    acf = t["acf"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(acf["lag"].astype(str), acf["acf"]); ax.set_title("Autocorrelation of total demand")
    save(fig, "autocorrelation")
    return files
