"""Tests for the data lifecycle: acquire -> clean -> audit -> integrate -> EDA -> CLI."""
import numpy as np
import pandas as pd
import pytest

from demand_agents import cli
from demand_agents.config import load_config
from demand_agents.data import acquire
from demand_agents.data.audit import audit_panel, audit_weather, demand_classes, mass_zero_days
from demand_agents.data.clean import clean_panel, clean_weather
from demand_agents.data.integrate import build_dataset, merge_weather_frame
from demand_agents.data.loaders import load_dataset
from demand_agents.data.synthetic import generate_synthetic
from demand_agents.eda.report import modelling_implications, write_report
from demand_agents.eda.tables import run_eda


@pytest.fixture
def panel():
    return generate_synthetic(n_stores=2, n_items=6, n_days=500, seed=3)


def _check(a, name):
    return [i for i in a["issues"] if i["check"] == name]


def test_audit_detects_injected_defects(panel):
    d = panel.copy()
    d["sales"] = d["sales"].astype(float)
    d = pd.concat([d, d.iloc[:5]]).reset_index(drop=True)            # duplicates
    d.loc[d.index[10], "sales"] = -4                                  # negative sales
    d = d[~((d["series_id"] == "S1_I01") & d["date"].between("2021-03-01", "2021-03-10"))]  # a gap
    closure = pd.Timestamp("2021-12-25")
    d.loc[d["date"] == closure, "sales"] = 0                          # store closure
    a = audit_panel(d, "D", "test")
    assert a["quality"]["duplicates"] >= 5 and a["quality"]["negative_sales"] >= 1
    assert a["quality"]["series_with_gaps"] >= 1 and a["quality"]["missing_periods"] >= 10
    assert "2021-12-25" in a["demand"]["mass_zero_days"] and _check(a, "store-closure days")
    assert {i["severity"] for i in a["issues"]} >= {"error", "warn", "info"}


def test_clean_panel_applies_and_logs_every_rule(panel):
    d = pd.concat([panel, panel.iloc[:7]]).reset_index(drop=True)
    d["sales"] = d["sales"].astype(float)
    d.loc[d.index[100], "sales"] = -3          # rows 0-6 are duplicated, so edit rows that are not
    d.loc[d.index[101], "sell_price"] = 0
    d.loc[d.index[102], "temp"] = 400
    d.loc[d.index[103], "sales"] = 4.4
    clean, log = clean_panel(d, "count")
    steps = {s["step"]: s["rows_affected"] for s in log}
    assert steps["drop duplicate (series, date) rows"] == 7
    assert steps["negative sales set to 0"] >= 1
    assert (clean["sales"] >= 0).all() and (clean["sales"] % 1 == 0).all()
    assert clean.duplicated(["series_id", "date"]).sum() == 0
    assert (clean["temp"].dropna().between(-60, 55)).all() and (clean["sell_price"].dropna() > 0).all()


def test_clean_weather_interpolates_short_gaps_only():
    dates = pd.date_range("2020-01-01", periods=30)
    t = np.linspace(0, 10, 30)
    w = pd.DataFrame({"date": dates, "state_id": "CA", "temp": t})
    w = w[~w["date"].isin(pd.to_datetime(["2020-01-05", "2020-01-06", "2020-01-15", "2020-01-16", "2020-01-17", "2020-01-18", "2020-01-19"]))]
    out, log = clean_weather(w, max_gap_days=3)
    assert len(out) == 30
    assert out.loc[out["date"] == "2020-01-05", "temp"].notna().all()           # 2-day gap filled
    assert out.loc[out["date"].between("2020-01-15", "2020-01-19"), "temp"].isna().any()   # 5-day gap stays missing


def test_demand_classes_follow_syntetos_boylan():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"series_id": np.repeat(["smooth", "intermittent"], 200), "date": list(pd.date_range("2020-01-01", periods=200)) * 2})
    d["sales"] = np.r_[rng.poisson(20, 200), (rng.random(200) < 0.1) * rng.poisson(5, 200) + (rng.random(200) < 0.1)]
    c = demand_classes(d).set_index("series_id")["class"]
    assert c["smooth"] == "smooth" and c["intermittent"] in ("intermittent", "lumpy")


def test_merge_weather_by_state_and_coverage_report():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4).tolist() * 2, "store_id": ["CA_1"] * 4 + ["TX_1"] * 4,
                       "series_id": "x", "sales": 1.0, "temp": np.nan})
    w = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "state_id": "CA", "temp": [1.0, 2.0, 3.0]})
    m = merge_weather_frame(df, w)
    assert m.loc[m["store_id"] == "CA_1", "temp"].tolist()[:3] == [1.0, 2.0, 3.0] and m.loc[m["store_id"] == "TX_1", "temp"].isna().all()
    rep = audit_weather(w, df["date"].unique(), states=["CA", "TX"])
    assert rep["states_without_weather"] == ["TX"] and rep["panel_date_coverage"] < 1.0
    assert any(i["check"] == "missing states" for i in rep["issues"])


def test_ghcn_parser_handles_tenths_of_degrees():
    csv = "STATION,DATE,TMAX,TMIN\nX,2020-01-01,250,100\nX,2020-01-02,,\nX,2020-01-03,300,150\n"
    out = acquire.parse_ghcn_daily(csv, "CA")
    assert out["temp"].tolist() == [17.5, 22.5] and set(out["state_id"]) == {"CA"}


def test_noaa_raw_folder_is_read_by_state_prefix(tmp_path):
    (tmp_path / "CA_USW1.csv").write_text("STATION,DATE,TMAX,TMIN\nX,2020-01-01,200,100\n")
    (tmp_path / "TX_USW2.csv").write_text("STATION,DATE,TMAX,TMIN\nX,2020-01-01,300,200\n")
    w = acquire.read_noaa_raw(tmp_path)
    assert set(w["state_id"]) == {"CA", "TX"} and len(w) == 2


def test_kaggle_download_builds_the_right_command(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(acquire.shutil, "which", lambda x: "/usr/bin/kaggle")
    monkeypatch.setattr(acquire, "kaggle_ready", lambda: True)
    monkeypatch.setattr(acquire.subprocess, "run", lambda cmd, check: calls.append(cmd) or (tmp_path / "m5").mkdir(exist_ok=True))
    with pytest.raises(RuntimeError, match="missing"):               # files never appear because we only recorded the command
        acquire.download_m5(tmp_path / "m5")
    assert calls[0][:5] == ["/usr/bin/kaggle", "competitions", "download", "-c", "m5-forecasting-accuracy"]


def test_kaggle_without_credentials_gives_a_helpful_message(monkeypatch):
    monkeypatch.setattr(acquire.shutil, "which", lambda x: "/usr/bin/kaggle")
    monkeypatch.setattr(acquire, "kaggle_ready", lambda: False)
    with pytest.raises(RuntimeError, match="Settings -> API"):
        acquire.download_walmart("/tmp/nonexistent_walmart_dir")


def test_build_dataset_writes_files_and_processed_table_is_reused(tmp_path):
    cfg = load_config(overrides={"data.synthetic.n_stores": 2, "data.synthetic.n_items": 4, "data.synthetic.n_days": 450,
                                 "data.processed_path": str(tmp_path / "proc.csv.gz")})
    s = build_dataset(cfg, interim_dir=tmp_path / "interim", reports_dir=tmp_path / "reports")
    assert (tmp_path / "proc.csv.gz").exists() and (tmp_path / "interim" / "synthetic_clean.csv.gz").exists()
    assert (tmp_path / "reports" / "tables" / "data_audit_synthetic.json").exists() and s["rows_final"] > 0
    again = load_dataset(cfg)                                          # reads the processed table
    assert len(again) == s["rows_final"] and pd.api.types.is_datetime64_any_dtype(again["date"])
    assert set(again["shock_type"].unique()) <= {"", "heatwave", "event", "surge"} and again["is_shock"].dtype == bool


def test_eda_tables_report_and_cli(tmp_path, capsys):
    cfg_overrides = ["data.synthetic.n_stores=2", "data.synthetic.n_items=6", "data.synthetic.n_days=500"]
    rep = str(tmp_path / "reports")
    cli.main(["build", "--reports", rep, *sum([["--set", o] for o in cfg_overrides], []), "--set", f"data.processed_path={tmp_path}/p.csv.gz"])
    cli.main(["eda", "--reports", rep, *sum([["--set", o] for o in cfg_overrides], []), "--set", f"data.processed_path={tmp_path}/p.csv.gz"])
    md = (tmp_path / "reports" / "data_audit_synthetic.md").read_text()
    for section in ("## 1. Overview", "## 2. Cleaning applied", "## 3. Issues found", "## 4. Integration", "## 5. Exploratory findings",
                    "## 6. What this means for modelling"):
        assert section in md
    assert "Strong weekly pattern" in md and len(list((tmp_path / "reports" / "figures").glob("*.png"))) >= 7
    out = capsys.readouterr().out
    assert "Built 'synthetic'" in out and "Wrote" in out


def test_modelling_implications_react_to_the_data(panel):
    t = run_eda(panel)
    a = audit_panel(panel, "D", "x")
    text = " ".join(modelling_implications(a, t))
    assert "weekly" in text and "Weather" in text.title() or "weather" in text
    assert t["abc_summary"]["volume_share"].sum() == pytest.approx(1.0)
    assert t["events"]["available"] and t["weather"]["available"]


def test_named_configs_and_cli_accepts_names(tmp_path):
    from demand_agents.config import NAMED, load_named

    for name in NAMED:
        cfg = load_named(name)
        assert cfg["data"]["processed_path"] and cfg["output_dir"]
    assert load_named("synthetic_calm")["data"]["synthetic"]["shocks"] is False
    assert load_named("walmart_synthetic")["data"]["source"] == "walmart_synthetic" and load_named("m5")["data"]["noaa"]["raw_dir"] == "data/raw/noaa"
    assert load_named("m5_christmas")["data"]["m5"]["end_offset_days"] == 138
    with pytest.raises(KeyError):
        load_named("nope")


def test_set_repo_url_updates_every_notebook(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    shutil.copy(root / "scripts" / "set_repo_url.py", tmp_path / "scripts")
    shutil.copytree(root / "notebooks", tmp_path / "notebooks")
    subprocess.run([sys.executable, str(tmp_path / "scripts" / "set_repo_url.py"), "https://github.com/alice/my-capstone.git"], check=True)
    for nb in (tmp_path / "notebooks").glob("*.ipynb"):
        text = nb.read_text()
        assert "alice/my-capstone" in text and "YOUR-USERNAME" not in text
