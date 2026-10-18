# Inventory and cost assumptions (to be justified by the Inventory Planning Lead)

The code contains **placeholder** values. They are reasonable starting points, not facts about any real retailer.
Replace them with figures you can cite, and say where each came from in the report.

| Parameter (`configs/base.yaml`) | Placeholder | How to justify it | Where the sensitivity is shown |
|---|---|---|---|
| `inventory.holding_rate_annual` | 0.25 | Annual cost of carrying inventory as a fraction of its value (capital, storage, shrink, obsolescence). Commonly quoted ranges for retail are roughly 15-30%; cite a source you can show. | cost-parameter grid (10% / 25% / 50%) |
| `inventory.stockout_cost_frac_of_price` | 0.5 | Cost of one lost sale as a fraction of price: at least the lost gross margin, plus goodwill. Use the margin of the product category and state your goodwill assumption. | cost-parameter grid (0.2 / 0.5 / 1.0) |
| `inventory.lead_time_days` | 2 | Time from order to delivery for the replenishment route you assume (store from regional DC, DC from supplier ...). | lead-time sweep (0-6 days) |
| `inventory.review_period_days` | 1 | How often orders are placed. | change the config |
| `inventory.service_level` | 0.95 | Target in-stock probability; often set per product class. | service-level sweep (80-99.5%) and per-class option `abc_service_levels` |
| `lead_time_delay_prob/max`, `batch_size`, `min_order_qty`, `fixed_order_cost`, `order_rule` | off / 1 / 0 / 0 / base_stock | Supplier reliability, case-pack size, minimum order, cost per purchase order. | what-if tab in the dashboard |

Notes
- The ranking of policies was stable across every sweep we ran on synthetic data, but **absolute costs depend strongly on these values**.
- For Walmart weekly data one unit is one dollar of sales volume (`inventory.unit_label`), so holding and stockout costs apply to dollars of stock, not physical units.
- The forecast model is built for a `forecast.horizon` look-ahead. The Planner requires `lead_time + review_period (+ buffer) <= horizon`.
