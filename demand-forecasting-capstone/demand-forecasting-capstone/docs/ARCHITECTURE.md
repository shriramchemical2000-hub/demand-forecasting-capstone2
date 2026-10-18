# Architecture

## Data flow

1. **Load** (`data/loaders.py`) - canonical long format: `date, store_id, item_id, series_id, sales, sell_price, is_event, temp`.
2. **Features + split** (Forecaster) - lagged sales (>= 7 days), rolling stats, calendar, price, weather;
   chronological split into `train | val_cal | val_check | test`.
3. **Forecast** (Forecaster) - one global gradient-boosting model (Poisson loss for counts; LightGBM optional). Candidates
   (model, blend with ETS, seasonal naive) are compared on the held-out validation window; the blend needs a 2% margin.
4. **Calibrate** (Uncertainty) - split-conformal intervals on `val_cal`; residuals scaled by `sqrt(y_hat + 1)`.
5. **Audit** (Critic) - checks coverage on the untouched `val_check` window. If coverage is below target minus
   tolerance, it sends a *revision request* with a proposed interval scale; the Uncertainty agent widens and the
   Critic re-checks (up to `critic.max_revision_rounds`).
6. **Finalize** (Uncertainty) - refits on all validation data, computes per-series reliability scores, and measures how much
   larger the error of P-period demand sums is than independent daily sigmas imply (inflation factor, used by the Planner).
7. **Risk** (Critic) - per series-period signals:
   - calendar event;
   - abnormal weather (|z| > 2 vs the training climate), only for series whose sales were found to react to temperature
     (sensitivity learned on training data, pooled across stores by item) and only in the demand-raising direction;
   - error drift: recent forecast error / validation error, computed only from errors already observable (lagged by the horizon);
   - price shock: a discount deeper than the 1% quantile of training prices.
   `risk = min(2.5, 1 + 0.25 event + 0.35 weather + 1.25 clip(drift_ratio - 1, 0, 1) + 0.2 price)`.
7b. **Widen** (Uncertainty) - published intervals use half-width x risk multiplier on flagged days, then adaptive conformal updating
   (a miss observed `horizon` periods ago nudges the interval scale up, a hit lets it drift down).
8. **Plan** (Planner) - `S_t = [nowcast_t x] sum_{P periods} forecast + z * sqrt(sum (sigma * risk)^2)`, order `max(0, S_t - inventory position)`.
   The optional nowcast factor (off by default) re-centres expected demand only when the demand observed over the last K periods is significantly
   (|z| > 2) away from the forecast: `1 + 0.7 (actual / forecast - 1)`, clipped to [0.8, 2.0].
9. **Evaluate** - forecast, interval, critic and inventory metrics written to `outputs/<run>/`.

## Daily and weekly mode

`data.frequency` (D / W), `data.season_period` (7 / 52) and `data.target_type` (count / continuous) make the same
pipeline run on daily unit sales (M5) or weekly dollar sales (Walmart). All "days" settings then mean periods.
Count data uses `sqrt(y_hat + 1)` as the noise scale and a Poisson loss; continuous data uses `y_hat` and squared error.
Run weekly mode with `--config config/walmart.yaml` (inherits everything else from `default.yaml`).

## Time conventions

- Order placed at the **end of day t** arrives at the **start of day t + 1 + L** (L = lead time).
- Protection interval `P = L + R` (R = review period). Forecast horizon `H = 7` must satisfy `P <= H`.
- Lost sales (no backorders). Holding cost = `holding_rate_annual / 365 x price` per unit-day;
  stockout cost = `stockout_cost_frac_of_price x price` per lost unit.

## Reasoning traces

`outputs/<run>/traces.jsonl`, one JSON object per line:

```json
{"step": 14, "ts": "...", "agent": "Critic", "kind": "decision",
 "content": "Calibration approved: ...", "data": {}}
```

Kinds: `thought`, `action`, `observation`, `message`, `revision_request`, `decision`.
Use `TraceLogger.log(agent, kind, content, data)` from any new agent.

## Adding a new agent

1. Subclass `BaseAgent` (`agents/base.py`), read/write fields on `PipelineState` (`agents/state.py`).
2. Log reasoning with `self.think / act / observe / decide`; message other agents with `state.send(...)`.
3. Call it from `orchestrator.run_pipeline` and add a test in `tests/test_agents.py`.
