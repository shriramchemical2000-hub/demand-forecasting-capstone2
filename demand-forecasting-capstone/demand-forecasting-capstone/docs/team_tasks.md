# Team task board

`[x]` = already implemented and tested in this repository (read it, do not redo it). `[ ]` = open. Everything still open needs **real data, a real decision, or a person**.

Names follow the roles of the first-draft submission; swap them if assignments change.

## First: the shared start (everyone, 30 minutes)
1. Read `README.md` and `docs/PROJECT_STRUCTURE.md`.
2. Open notebook 02 -> 06 in Colab (or locally) on `DATASET = "synthetic"` and run them in order.
3. One person completes `docs/GITHUB_COLAB_SETUP.md` so the whole team can open the notebooks from the repository.

## Suvigya Gangele - Data Engineering Lead  (stages 1-3: `data/acquire.py`, `loaders.py`, `clean.py`, `audit.py`, `integrate.py`; notebooks 01-02)
- [x] Download, clean, audit and integrate code with tests; written cleaning rules (`docs/DATA.md`); M5 + NOAA merge by state
- [ ] Run notebook 01 for M5, Walmart and NOAA; then notebook 02 with `DATASET = "m5"` and `"walmart"`; fix anything the real files reveal
- [ ] Read `reports/data_audit_m5.md` and `..._walmart.md`: write the Data section of the report (sources, pilot subset, issues found and what was done)
- [ ] Justify the M5 pilot (which stores and items) and the NOAA stations; check weather coverage in the audit

## Krishna Kumar - Forecasting Lead  (stages 4-6: `eda/`, `data/features.py`, `models/`, `agents/forecaster.py`; notebooks 03-04)
- [x] EDA tables, plots and report; leakage-safe features; gradient boosting, LightGBM option, ETS/ARIMA baselines, model selection incl. ETS blend
- [ ] Write the EDA findings for the report from `reports/data_audit_<name>.md` (what the patterns mean for the model)
- [ ] Tune the model with time-based cross-validation on real data; try per-category models
- [ ] Report WAPE / RMSE / bias on real data, overall and on shock days

## Shriram P - Uncertainty Lead  (`models/uncertainty.py`, `agents/uncertainty.py`; notebook 05)
- [x] Conformal intervals (count / continuous), per-series shrinkage, risk-aware widening, adaptive updating, multi-period sigma correction, reliability scores, calibration curves and coverage by store / item
- [ ] Intervals under-cover during sudden unobserved surges (errors become visible only after the horizon): try quantile gradient boosting or a shorter-horizon model
- [ ] Study the real-data calibration plots and explain differences from the synthetic ones

## Sonali Pattanaik - Critic and Evaluation Lead  (`agents/critic.py`, `evaluation/`, `experiments.py`; notebook 06)
- [x] Learned weather sensitivity, learned risk weights, drift / event / price signals, precision and recall, ablation ladder, multi-seed suite, sweeps, equal-fill-rate comparison, trace-review export
- [ ] Re-check the Critic on real validation windows; add signals only if validation errors support them
- [ ] Organise the qualitative review (two raters, `docs/trace_review.md`) and quote traces in the report

## Bhaskara Veera Kumar Rayavarapu - Inventory Planning Lead  (`inventory/`, `agents/planner.py`)
- [x] Base-stock and (s, S) policies, optional nowcast, random supplier delays, batches / MOQ, fixed order cost, ABC service levels, lead-time buffer, what-if tab
- [ ] **Justify and cite the cost parameters and lead time** (`docs/ASSUMPTIONS.md` lists what to fill in)
- [ ] Decide how to present Walmart "units" (dollars of sales)

## Dilip S Biradar - Integration and Delivery Lead  (`orchestrator.py`, `cli.py`, `dashboard/`, repository, report)
- [x] Orchestrator + revision loop, command line, dashboard, six notebooks, CI workflow, `CONTRIBUTING.md`
- [ ] Create the GitHub repository, push, confirm the CI is green and the Colab badges open
- [ ] Assemble the final report (`reports/report_draft_<name>.md` plus the team's sections) and record the demo video

## Everyone
- Choose settings on validation data; report the test window once.
- Report real-data and synthetic results separately. Compare policies at equal fill rate.
- Be able to explain your own part and the whole flow in the viva; read `docs/LIMITATIONS.md`.
