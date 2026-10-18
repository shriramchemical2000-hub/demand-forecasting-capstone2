"""Assemble result tables and write all outputs (CSV / JSON) used by the dashboard and the report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .metrics import bias, rmse, wape
from ..models.uncertainty import coverage, noise_scale


def assemble_forecast_table(state) -> pd.DataFrame:
    f = state.forecasts.copy()
    f = f.merge(state.risk, on=["date", "series_id"], how="left")
    for name, b in state.baselines.items():
        f = f.merge(b.rename(columns={"y_hat": name}), on=["date", "series_id"], how="left")
    extras = [c for c in ["category", "shock_type", "is_shock", "temp"] if c in state.panel.columns]
    f = f.merge(state.panel[["date", "series_id"] + extras], on=["date", "series_id"], how="left")
    return f.sort_values(["series_id", "date"]).reset_index(drop=True)


def critic_flagged(test: pd.DataFrame) -> pd.Series:
    """A series-period counts as flagged when the Critic raised any discrete shock flag."""
    flags = test["event_flag"] | test["weather_flag"] | test["drift_flag"]
    if "price_flag" in test.columns:
        flags = flags | test["price_flag"].fillna(False).astype(bool)
    return flags.astype(bool)


def _shock_mask(test: pd.DataFrame) -> pd.Series:
    if "is_shock" in test.columns:
        return test["is_shock"].astype(bool)
    return critic_flagged(test)


def forecast_metrics(test: pd.DataFrame, models: dict) -> pd.DataFrame:
    shock = _shock_mask(test)
    segments = {"all": pd.Series(True, index=test.index), "shock_days": shock, "normal_days": ~shock}
    rows = []
    for label, col in models.items():
        for seg, mask in segments.items():
            d = test[mask & test[col].notna()]
            if len(d) == 0:
                continue
            rows.append(
                {"model": label, "segment": seg, "n": len(d), "wape": wape(d["sales"], d[col]),
                 "rmse": rmse(d["sales"], d[col]), "bias": bias(d["sales"], d[col])}
            )
    return pd.DataFrame(rows)


def interval_metrics(test: pd.DataFrame) -> pd.DataFrame:
    shock = _shock_mask(test)
    rows = []
    for seg, mask in {"all": pd.Series(True, index=test.index), "shock_days": shock, "normal_days": ~shock}.items():
        d = test[mask]
        if len(d):
            rows.append({"segment": seg, "n": len(d), "coverage": coverage(d["sales"], d["lower"], d["upper"]),
                         "mean_width": float((d["upper"] - d["lower"]).mean())})
    if "shock_type" in test.columns:  # break the shock days down by cause (synthetic data has ground truth)
        for kind in sorted(set(test["shock_type"].dropna()) - {""}):
            d = test[test["shock_type"] == kind]
            rows.append({"segment": f"shock:{kind}", "n": len(d), "coverage": coverage(d["sales"], d["lower"], d["upper"]),
                         "mean_width": float((d["upper"] - d["lower"]).mean())})
    return pd.DataFrame(rows)


def critic_metrics(test: pd.DataFrame) -> dict:
    flagged = critic_flagged(test)
    kind = "count" if (test["y_hat"].max() < 1e4) else "continuous"
    nz = (test["sales"] - test["y_hat"]).abs() / noise_scale(test["y_hat"], kind)
    out = {
        "flagged_share": float(flagged.mean()),
        "mean_abs_error_flagged": float(nz[flagged].mean()) if flagged.any() else None,
        "mean_abs_error_unflagged": float(nz[~flagged].mean()) if (~flagged).any() else None,
    }
    if out["mean_abs_error_flagged"] and out["mean_abs_error_unflagged"]:
        out["error_ratio_flagged_vs_unflagged"] = out["mean_abs_error_flagged"] / out["mean_abs_error_unflagged"]
    if "is_shock" in test.columns:
        truth = test["is_shock"].astype(bool)
        out["precision"] = float(truth[flagged].mean()) if flagged.any() else None
        out["recall"] = float(flagged[truth].mean()) if truth.any() else None
    return out


def psum_calibration(test: pd.DataFrame, P: int, kappa: float) -> dict:
    """How well does sqrt(sum sigma^2) describe the error of demand summed over the protection interval?"""
    t = test.sort_values(["series_id", "date"])
    err = (t["sales"] - t["y_hat"]).groupby(t["series_id"]).transform(lambda x: x.rolling(P).sum())
    var = (t["sigma"] ** 2).groupby(t["series_id"]).transform(lambda x: x.rolling(P).sum())
    z = (err / np.sqrt(var)).dropna()
    return {
        "P": P, "kappa": kappa,
        "z_std_raw": float(z.std()), "z_std_after_inflation": float(z.std() / kappa),
        "coverage_90_raw": float((z.abs() <= 1.645).mean()),
        "coverage_90_after_inflation": float(((z / kappa).abs() <= 1.645).mean()),
    }


def interval_by_group(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in ("store_id", "item_id"):
        for g, d in test.groupby(key):
            rows.append({"group_by": key, "group": g, "n": len(d),
                         "coverage": coverage(d["sales"], d["lower"], d["upper"]),
                         "coverage_base_intervals": coverage(d["sales"], d["lower_base"], d["upper_base"]),
                         "mean_width": float((d["upper"] - d["lower"]).mean())})
    return pd.DataFrame(rows)


def calibration_curve(table: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reliability diagram data: nominal vs empirical coverage of the *base* intervals (before risk widening),
    using quantiles of validation z-scores, evaluated on the test window."""
    val = table[table["split"].isin(["val_cal", "val_check"])]
    test = table[table["split"] == "test"]
    zv = ((val["sales"] - val["y_hat"]) / noise_scale(val["y_hat"], kind)).abs()
    zt = ((test["sales"] - test["y_hat"]) / noise_scale(test["y_hat"], kind)).abs()
    shock = _shock_mask(test)
    rows = []
    for level in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        q = float(zv.quantile(level))
        for seg, mask in (("all", pd.Series(True, index=test.index)), ("shock_days", shock), ("normal_days", ~shock)):
            if mask.any():
                rows.append({"nominal": level, "segment": seg, "empirical": float((zt[mask] <= q).mean())})
    return pd.DataFrame(rows)


def evaluate_and_save(state, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = assemble_forecast_table(state)
    test = table[table["split"] == "test"]

    models = {"agentic_forecast": "y_hat", "gbm_only": "gbm", "seasonal_naive": "seasonal_naive"}
    for name in state.baselines:
        models[name] = name
    fm = forecast_metrics(test, models)
    im = interval_metrics(test)
    cm = critic_metrics(test)

    inv_cfg = state.cfg["inventory"]
    P = inv_cfg["lead_time_days"] + inv_cfg["review_period_days"]
    kind = state.cfg["data"].get("target_type", "count")
    psum = psum_calibration(test, P, state.psum_inflation)
    by_group = interval_by_group(test)
    curve = calibration_curve(table, kind)
    by_group.to_csv(out / "interval_by_group.csv", index=False)
    curve.to_csv(out / "calibration_curve.csv", index=False)
    table.to_csv(out / "forecasts.csv", index=False)
    if state.price is not None:
        state.price.rename("price").rename_axis("series_id").to_csv(out / "prices.csv")
    with open(out / "baseline_sigma.json", "w") as f:
        json.dump(state.baseline_sigma, f)
    fm.to_csv(out / "forecast_metrics.csv", index=False)
    im.to_csv(out / "interval_metrics.csv", index=False)
    state.inventory_summary.to_csv(out / "inventory_summary.csv", index=False)
    state.inventory_daily.to_csv(out / "inventory_daily.csv", index=False)
    state.recommendations.to_csv(out / "recommendations.csv", index=False)
    if state.reliability is not None:
        state.reliability.to_csv(out / "reliability.csv", index=False)
    with open(out / "config_used.yaml", "w") as f:
        yaml.safe_dump(state.cfg, f, sort_keys=False)

    summary = {
        "splits": {k: str(v.date()) for k, v in state.splits.items()},
        "n_series": int(test["series_id"].nunique()),
        "calibration_checks": state.metrics.get("calibration_checks", []),
        "interval_scale": state.interval_scale,
        "critic": cm,
        "forecaster": state.metrics.get("forecaster", {}),
        "psum_calibration": psum,
        "risk_weights": state.metrics.get("risk_weights", {}),
        "unit_label": state.cfg["inventory"].get("unit_label", "units"),
        "psum_inflation": state.psum_inflation,
        "forecast_metrics": fm.to_dict(orient="records"),
        "interval_metrics": im.to_dict(orient="records"),
        "inventory_summary": state.inventory_summary.to_dict(orient="records"),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    state.metrics.update({"summary": summary})
    return summary
