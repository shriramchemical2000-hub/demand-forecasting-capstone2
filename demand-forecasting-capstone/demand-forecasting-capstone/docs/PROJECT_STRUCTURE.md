# Project structure

```
demand-forecasting-capstone/
├── README.md
├── pyproject.toml              installable package (`pip install -e .`) and the `demand-agents` command
├── configs/                    settings: base.yaml (synthetic), m5.yaml, m5_christmas.yaml, walmart.yaml
├── data/
│   ├── raw/                    downloaded files, untouched            (git-ignored)
│   ├── interim/                cleaned per-dataset tables             (git-ignored)
│   └── processed/              the modelling tables the models read   (git-ignored)
├── notebooks/                  01 ... 06, one per lifecycle stage
├── src/demand_agents/
│   ├── cli.py                  demand-agents download | build | eda | run
│   ├── config.py               loads configs; dataset names (m5, walmart, ...)
│   ├── data/
│   │   ├── acquire.py          1  download Kaggle / NOAA data
│   │   ├── loaders.py          1  raw files -> one common table format
│   │   ├── clean.py            2  documented cleaning rules (each counted and logged)
│   │   ├── audit.py            2  data-quality measurements and recommended actions
│   │   ├── integrate.py        3  clean -> audit -> merge weather -> data/processed
│   │   ├── features.py         5  leakage-safe model inputs
│   │   ├── splits.py           5  train / validation / test in time order
│   │   └── synthetic.py           practice data with known shocks (for tests and demos)
│   ├── eda/                    4  tables, plots and the written report
│   ├── models/                 6  gradient boosting, classical baselines, conformal intervals
│   ├── agents/                 6  forecaster, uncertainty, critic, planner
│   ├── inventory/              6  reorder policy and simulator
│   ├── evaluation/             7  metrics and result tables
│   ├── orchestrator.py         6  runs the agents in order
│   ├── experiments.py          7  fair-comparison sweeps and multi-seed runs
│   └── tracing.py                 reasoning traces (JSONL)
├── scripts/                    report figures, draft report, trace-review sheet, set_repo_url
├── dashboard/app.py            Streamlit dashboard
├── reports/                    COMMITTED results: data_audit_<name>.md, figures/, tables/
├── outputs/                    run artefacts per dataset (git-ignored)
├── tests/                      automated tests
└── docs/                       DATA, ARCHITECTURE, ASSUMPTIONS, LIMITATIONS, GITHUB_COLAB_SETUP, team_tasks, trace_review
```

## What creates what

| Command or notebook | Reads | Writes |
|---|---|---|
| `demand-agents download` / nb 01 | the internet | `data/raw/` |
| `demand-agents build` / nb 02 | `data/raw/` (or generates synthetic) | `data/interim/`, `data/processed/`, `reports/tables/data_audit_<name>.json`, `cleaning_log_<name>.csv` |
| `demand-agents eda` / nb 03 | `data/processed/` | `reports/data_audit_<name>.md`, `reports/figures/` |
| `demand-agents run` / nb 05 | `data/processed/` | `outputs/<name>/` (forecasts, metrics, inventory, traces, `data_audit.json`) |
| notebook 06 | `outputs/<name>/` | `outputs/experiments_<name>/`, `reports/figures/<name>/`, `reports/report_draft_<name>.md` |

`data/processed/` is used automatically when it exists; if you change a data setting, run `build` again.
