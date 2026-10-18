# Agentic Demand Forecasting & Inventory Optimization

A capstone project: a small team of cooperating AI "agents" forecasts demand, measures how uncertain each forecast is, flags risky days, and
recommends how much stock to hold, and it is compared fairly with classical forecasting methods.

**Problem.** Demand forecasts often fail during demand shocks (holidays, weather, sudden surges). **Objectives:** improve forecast robustness,
optimise reorder policies, reduce stockouts and overstock, and compare with classical models.
**Data:** M5 Forecasting (Kaggle), Walmart sales (Kaggle) and NOAA weather. See [`docs/DATA.md`](docs/DATA.md).

## The project follows the data science lifecycle

```
 1 Acquire  ->  2 Audit & clean  ->  3 Integrate  ->  4 EDA  ->  5 Features & split  ->  6 Agents & models  ->  7 Evaluate & report
 notebook 01     notebook 02          notebook 02       nb 03        notebook 04            notebook 05            notebook 06
 data/raw/       data/interim/        data/processed/   reports/     (in memory)            outputs/               reports/
```

| Notebook | What it does |
|---|---|
| `01_data_acquisition` | downloads M5, Walmart (Kaggle) and NOAA weather into `data/raw/` |
| `02_data_audit_and_cleaning` | measures data quality, applies written cleaning rules, builds the modelling table |
| `03_eda` | charts and findings: seasonality, events, price, weather, demand types; writes `reports/data_audit_<name>.md` |
| `04_integration_and_features` | which datasets are combined and why; leakage-safe features; time split |
| `05_agents_and_modelling` | runs the four agents and shows their reasoning traces |
| `06_evaluation_and_report` | fair comparison with classical methods, sensitivity tests, figures and a draft report |

## Run it - no uploading of files

**Google Colab (recommended for submission).** The notebooks download the code from GitHub and the data from Kaggle/NOAA themselves.
One-time setup in [`docs/GITHUB_COLAB_SETUP.md`](docs/GITHUB_COLAB_SETUP.md). After that: open a notebook from GitHub in Colab, `Runtime -> Run all`.

**On your computer.**

```bash
pip install -e ".[dev,dashboard]"
demand-agents build --config synthetic      # clean + audit + build the modelling table (no downloads needed)
demand-agents eda   --config synthetic      # data audit + EDA report in reports/
demand-agents run   --config synthetic      # the agents -> outputs/synthetic/
streamlit run dashboard/app.py              # explore the results
pytest -q                                   # tests
```

`--config` takes a dataset name: `synthetic`, `synthetic_calm`, `m5`, `m5_christmas`, `walmart`, `walmart_synthetic`.
For real data run `demand-agents download --dataset all` first (needs a free Kaggle token).

## Layout

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). In short: `src/demand_agents/` (the code, by lifecycle stage), `configs/` (settings),
`data/` (raw -> interim -> processed, never committed), `notebooks/` (the story), `reports/` (committed results), `outputs/` (run artefacts), `tests/`, `docs/`.

## Results

None are stored in this README on purpose: they are generated from your data by notebook 06 into `reports/`.
Read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before presenting anything, and compare policies **at equal fill rate**, never at one setting.

## Team

Tasks per person: [`docs/team_tasks.md`](docs/team_tasks.md). Working rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).
