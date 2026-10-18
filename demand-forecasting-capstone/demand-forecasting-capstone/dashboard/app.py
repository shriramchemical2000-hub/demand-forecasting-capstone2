"""Demo dashboard. Run:  streamlit run dashboard/app.py   (after python scripts/run_pipeline.py)"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Agentic Demand Forecasting", layout="wide")
st.title("Agentic Demand Forecasting & Inventory Optimization")

_found = sorted((ROOT / "outputs").glob("*/metrics.json"), key=lambda f: f.stat().st_mtime)
_default = _found[-1].parent if _found else ROOT / "outputs" / "synthetic"
out_dir = Path(st.sidebar.text_input("Results folder", str(_default)))
if not (out_dir / "metrics.json").exists():
    st.warning(f"No results found in `{out_dir}`. Run `demand-agents run` (or notebook 05) first.")
    st.stop()


@st.cache_data
def load(path: str):
    p = Path(path)
    data = {
        "metrics": json.loads((p / "metrics.json").read_text()),
        "forecasts": pd.read_csv(p / "forecasts.csv", parse_dates=["date"]),
        "fm": pd.read_csv(p / "forecast_metrics.csv"),
        "im": pd.read_csv(p / "interval_metrics.csv"),
        "inv": pd.read_csv(p / "inventory_summary.csv"),
        "daily": pd.read_csv(p / "inventory_daily.csv", parse_dates=["date"]),
        "recs": pd.read_csv(p / "recommendations.csv"),
        "traces": pd.read_json(p / "traces.jsonl", lines=True),
        "curve": pd.read_csv(p / "calibration_curve.csv") if (p / "calibration_curve.csv").exists() else None,
        "groups": pd.read_csv(p / "interval_by_group.csv") if (p / "interval_by_group.csv").exists() else None,
    }
    return data


d = load(str(out_dir))
fc, daily = d["forecasts"], d["daily"]
test = fc[fc["split"] == "test"]

unit = d["metrics"].get("unit_label", "units")
tab_over, tab_fc, tab_cal, tab_inv, tab_what, tab_rec, tab_exp, tab_trace = st.tabs(
    ["Overview", "Forecasts", "Calibration", "Inventory simulation", "What-if planner", "Order recommendations",
     "Experiments", "Agent reasoning traces"]
)

# ------------------------------------------------------------------ overview
with tab_over:
    wape_all = d["fm"][d["fm"]["segment"] == "all"].set_index("model")["wape"]
    inv = d["inv"].set_index("policy")
    classical = next((p for p in inv.index if p.startswith("classical")), inv.index[0])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WAPE (agentic forecast)", f"{wape_all.get('agentic_forecast', float('nan')):.3f}",
              f"{wape_all.get('agentic_forecast', 0) - wape_all.get('ets', wape_all.get('seasonal_naive', 0)):+.3f} vs classical",
              delta_color="inverse")
    c2.metric("Fill rate (agentic)", f"{inv.loc['agentic', 'fill_rate']:.1%}",
              f"{inv.loc['agentic', 'fill_rate'] - inv.loc[classical, 'fill_rate']:+.1%} vs classical")
    c3.metric("Stockout rate (agentic)", f"{inv.loc['agentic', 'stockout_rate']:.1%}",
              f"{inv.loc['agentic', 'stockout_rate'] - inv.loc[classical, 'stockout_rate']:+.1%} vs classical",
              delta_color="inverse")
    c4.metric("Total cost (agentic)", f"{inv.loc['agentic', 'total_cost']:,.0f}",
              f"{inv.loc['agentic', 'total_cost'] / inv.loc[classical, 'total_cost'] - 1:+.0%} vs classical",
              delta_color="inverse")

    st.subheader("Forecast error by segment (WAPE, lower is better)")
    pivot = d["fm"].pivot(index="model", columns="segment", values="wape").round(3)
    st.dataframe(pivot, width="stretch")
    fig = go.Figure()
    for seg in pivot.columns:
        fig.add_bar(x=pivot.index, y=pivot[seg], name=seg)
    fig.update_layout(barmode="group", height=320, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Prediction-interval coverage (test)")
        st.dataframe(d["im"].round(3), width="stretch", hide_index=True)
    with right:
        st.subheader("Critic quality")
        st.json(d["metrics"].get("critic", {}))

# ------------------------------------------------------------------ forecasts
with tab_fc:
    sid = st.selectbox("Store-item", sorted(test["series_id"].unique()))
    extra = [c for c in ["seasonal_naive", "ets", "arima"] if c in test.columns]
    show = st.multiselect("Baselines to overlay", extra, default=[c for c in extra if c == "ets"])
    s = fc[(fc["series_id"] == sid) & (fc["split"].isin(["val_check", "test"]))]
    fig = go.Figure()
    fig.add_scatter(x=s["date"], y=s["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=s["date"], y=s["lower"], fill="tonexty", line=dict(width=0),
                    fillcolor="rgba(70,130,180,0.2)", name="90% interval")
    fig.add_scatter(x=s["date"], y=s["sales"], name="Actual", mode="lines+markers", marker=dict(size=4),
                    line=dict(color="black", width=1))
    fig.add_scatter(x=s["date"], y=s["y_hat"], name="Agentic forecast", line=dict(color="steelblue"))
    for b in show:
        fig.add_scatter(x=s["date"], y=s[b], name=b, line=dict(dash="dot"))
    flagged = s[s["risk_multiplier"] > 1]
    fig.add_scatter(x=flagged["date"], y=flagged["sales"], mode="markers", name="Critic flag",
                    marker=dict(color="crimson", size=9, symbol="diamond-open"))
    fig.add_vline(x=pd.Timestamp(d["metrics"]["splits"]["test_start"]), line_dash="dash", line_color="gray")
    fig.update_layout(height=450, margin=dict(t=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    if "shock_type" in s.columns:
        st.caption("Ground-truth shocks (synthetic data): " + ", ".join(sorted(set(s["shock_type"].dropna()) - {""})) or "none")

# ------------------------------------------------------------------ calibration
with tab_cal:
    st.subheader("Reliability diagram (base intervals, test window)")
    st.caption("A well-calibrated forecaster lies on the diagonal. Normal days should; shock days show what the Critic and "
               "the risk-widening have to fix.")
    if d["curve"] is not None:
        fig = go.Figure()
        fig.add_scatter(x=[0.5, 0.95], y=[0.5, 0.95], mode="lines", line=dict(dash="dash", color="gray"), name="ideal")
        for seg, g in d["curve"].groupby("segment"):
            fig.add_scatter(x=g["nominal"], y=g["empirical"], mode="lines+markers", name=seg)
        fig.update_layout(height=380, xaxis_title="nominal coverage", yaxis_title="empirical coverage", margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    psum = d["metrics"].get("psum_calibration")
    if psum:
        st.subheader("Demand over the protection interval")
        st.write(f"Standardised error of {psum['P']}-period demand sums has std {psum['z_std_raw']:.2f} on the test window "
                 f"(1.0 would mean the independent-error sigma is right). Planner sigma is inflated by x{psum['kappa']:.2f}, "
                 "estimated on validation data.")
    if d["groups"] is not None:
        st.subheader("Coverage by store and by item (90% target)")
        c1, c2 = st.columns(2)
        for col, key in ((c1, "store_id"), (c2, "item_id")):
            g = d["groups"][d["groups"]["group_by"] == key].sort_values("group")
            f = go.Figure(go.Bar(x=g["group"], y=g["coverage"]))
            f.add_hline(y=0.9, line_dash="dash")
            f.update_layout(title=key, height=300, margin=dict(t=40), yaxis_range=[0.5, 1.0])
            col.plotly_chart(f, width="stretch")

# ------------------------------------------------------------------ what-if planner
with tab_what:
    st.subheader("Re-plan with your own assumptions (forecasts stay fixed)")
    st.caption(f"One unit = {unit}. The Planner re-runs on the saved forecasts, so this takes a moment, not a full pipeline run.")
    cfg_inv = yaml_cfg = None
    try:
        import yaml
        from demand_agents.experiments import load_state_from_outputs, replan

        base_cfg = yaml.safe_load((out_dir / "config_used.yaml").read_text())["inventory"]
        H = yaml.safe_load((out_dir / "config_used.yaml").read_text())["forecast"]["horizon"]
        c1, c2, c3 = st.columns(3)
        service = c1.slider("Service level", 0.80, 0.995, float(base_cfg["service_level"]), 0.005)
        lead = c2.slider("Lead time", 0, max(0, H - 1 - base_cfg.get("lead_time_buffer", 0)), int(base_cfg["lead_time_days"]))
        hold = c3.slider("Holding cost per year (fraction of value)", 0.05, 0.60, float(base_cfg["holding_rate_annual"]), 0.05)
        c4, c5, c6 = st.columns(3)
        stock = c4.slider("Stockout cost (fraction of price)", 0.1, 2.0, float(base_cfg["stockout_cost_frac_of_price"]), 0.1)
        delay = c5.slider("Chance an order arrives late", 0.0, 0.6, float(base_cfg.get("lead_time_delay_prob", 0.0)), 0.05)
        batch = c6.number_input("Order batch size", 1, 500, int(base_cfg.get("batch_size", 1)))
        overrides = {"service_level": service, "lead_time_days": lead, "holding_rate_annual": hold,
                     "stockout_cost_frac_of_price": stock, "lead_time_delay_prob": delay,
                     "lead_time_delay_max": max(1, int(base_cfg.get("lead_time_delay_max", 0)) or 2) if delay > 0 else 0,
                     "batch_size": int(batch)}
        if lead + 1 + base_cfg.get("lead_time_buffer", 0) > H:
            st.error(f"Lead time + review period must not exceed the forecast horizon ({H}).")
        else:
            res = replan(load_state_from_outputs(out_dir), overrides)
            st.dataframe(res.round(3), width="stretch", hide_index=True)
            fig = go.Figure(go.Bar(x=res["policy"], y=res["total_cost"]))
            fig.update_layout(title="Total cost", height=300, margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")
    except FileNotFoundError as e:
        st.info(f"This results folder was created by an older version (missing {Path(str(e)).name}). Re-run the pipeline.")

# ------------------------------------------------------------------ inventory
with tab_inv:
    st.subheader("Policy comparison (same test window, same demand)")
    st.caption(f"One unit = {unit}.")
    st.dataframe(d["inv"].round(3), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    for col, metric, title in [(c1, "fill_rate", "Fill rate"), (c2, "total_cost", "Total cost")]:
        f = go.Figure(go.Bar(x=d["inv"]["policy"], y=d["inv"][metric]))
        f.update_layout(title=title, height=300, margin=dict(t=40))
        col.plotly_chart(f, width="stretch")

    st.subheader("Stock level for one store-item")
    sid2 = st.selectbox("Store-item ", sorted(daily["series_id"].unique()), key="inv_sid")
    pol = st.selectbox("Policy", sorted(daily["policy"].unique()), index=sorted(daily["policy"].unique()).index("agentic"))
    s = daily[(daily["series_id"] == sid2) & (daily["policy"] == pol)]
    fig = go.Figure()
    fig.add_scatter(x=s["date"], y=s["on_hand_end"], name="On hand (end of day)", fill="tozeroy")
    fig.add_scatter(x=s["date"], y=s["order_up_to"], name="Order-up-to level", line=dict(dash="dot"))
    fig.add_bar(x=s["date"], y=s["lost"], name="Lost sales", marker_color="crimson")
    fig.update_layout(height=380, margin=dict(t=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ recommendations
with tab_rec:
    st.subheader("Order-up-to recommendations for the next planning cycle")
    recs = d["recs"].sort_values("risk_multiplier", ascending=False)
    st.dataframe(recs.round(2), width="stretch", hide_index=True)

# ------------------------------------------------------------------ experiments
with tab_exp:
    exp_dir = Path(st.text_input("Experiments folder", str(ROOT / "outputs" / f"experiments_{out_dir.name}")))
    st.caption("Created by notebook 06 (or `python scripts/run_experiments.py`).")
    seeds_file = exp_dir.parent / "experiments" / "seeds" / "seeds_summary.csv"
    if seeds_file.exists():
        st.subheader("Robustness: mean and std over independent synthetic worlds")
        st.dataframe(pd.read_csv(seeds_file).round(3), width="stretch", hide_index=True)
    for name, title, y in [
        ("service_level", "Service-level sweep: total cost", "total_cost"),
        ("lead_time", "Lead-time sweep: total cost", "total_cost"),
    ]:
        f = exp_dir / f"sensitivity_{name}.csv"
        if f.exists():
            df = pd.read_csv(f)
            x = "service_level" if name == "service_level" else "lead_time"
            st.subheader(title)
            fig = go.Figure()
            for pol, g in df.groupby("policy"):
                fig.add_scatter(x=g[x], y=g[y], mode="lines+markers", name=pol)
            fig.update_layout(height=340, margin=dict(t=20), xaxis_title=x, yaxis_title=y)
            st.plotly_chart(fig, width="stretch")
    f = exp_dir / "sensitivity_frontier.csv"
    if f.exists():
        st.subheader("Fair comparison: average stock needed to reach a target fill rate (lower is better)")
        st.dataframe(pd.read_csv(f).pivot(index="target_fill_rate", columns="policy", values="avg_stock_needed").round(1),
                     width="stretch")
    f = exp_dir / "sensitivity_cost_params.csv"
    if f.exists():
        st.subheader("Cost-parameter grid: total cost")
        df = pd.read_csv(f)
        st.dataframe(df.pivot_table(index=["holding_rate_annual", "stockout_cost_frac"], columns="policy",
                                    values="total_cost").round(0), width="stretch")
    if not seeds_file.exists() and not f.exists():
        st.info("No experiment results found yet.")

# ------------------------------------------------------------------ traces
with tab_trace:
    tr = d["traces"]
    a, k = st.columns(2)
    agents = a.multiselect("Agents", sorted(tr["agent"].unique()), default=sorted(tr["agent"].unique()))
    kinds = k.multiselect("Event types", sorted(tr["kind"].unique()), default=sorted(tr["kind"].unique()))
    view = tr[tr["agent"].isin(agents) & tr["kind"].isin(kinds)]
    for _, e in view.iterrows():
        with st.expander(f"#{e['step']}  {e['agent']}  ·  {e['kind']}  ·  {e['content'][:90]}"):
            st.write(e["content"])
            if e["data"]:
                st.json(e["data"])
