# Data

## Sources

| Dataset | What it is | Access |
|---|---|---|
| **M5 Forecasting - Accuracy** | Daily unit sales of Walmart products: 3 states (CA, TX, WI), 10 stores, 3,049 items (30,490 item-store series), 29 Jan 2011 - 19 Jun 2016, with a calendar (events) and weekly prices | Kaggle competition `m5-forecasting-accuracy`. Create a Kaggle token (Settings -> API) **and** click *Join / Late Submission* on the competition page once |
| **Walmart sales** | Weekly sales of 45 stores, 5 Feb 2010 - 26 Oct 2012, with holiday flag, temperature, fuel price, CPI, unemployment | Kaggle dataset `yasserh/walmart-dataset` (token only) |
| **NOAA weather** | Daily temperature (GHCN-Daily station files) | public, no key: ncei.noaa.gov |

Raw files are **never committed to git**. `demand-agents download` (or notebook 01) fetches them.
M5 pilot subset (configurable in `configs/base.yaml`): first 3 stores, 50 highest-volume items, latest 1,000 days.

## Common table format

Every dataset is converted to one long table: `date, store_id, item_id, series_id, sales, sell_price, is_event, temp`
(+ extra columns such as `fuel_price`; synthetic data also has `shock_type`). One row = one series in one period.

## Cleaning rules (stage 2)

Each rule is applied by `data/clean.py`, counted, and written to `reports/tables/cleaning_log_<name>.csv`.

| Rule | Why |
|---|---|
| rows with missing date, series or sales are dropped | cannot be used |
| duplicate (series, date) rows: keep the last | one observation per series and day |
| negative sales set to 0 | returns are not demand |
| non-integer unit sales rounded | count data must be whole numbers |
| price <= 0 set to missing, then carried forward within the series | a price cannot be zero; prices change weekly, not daily |
| temperature outside -60..55 C set to missing | sensor errors |
| weather gaps of <= 3 days interpolated; longer gaps stay missing | never invent long stretches of weather |
| M5: rows before an item's first priced week are dropped | the item was not on sale yet; those zeros are not demand |

**Deliberately not done:** removing demand spikes or zero-sales days. Spikes are the shocks this project studies; zeros are real
(slow items, store closures such as Christmas Day).

## Integration decisions (stage 3)

| Combination | Decision | Reason |
|---|---|---|
| M5 sales + calendar + prices | one table | same items, stores, days (done in the M5 loader) |
| M5 + NOAA | merged by state and date | weather is external; M5 hides store locations, so one representative station per state is used (CA: Los Angeles, TX: Houston, WI: Milwaukee) |
| Walmart + M5 | **not merged** | different stores, period, measure (weekly dollars vs daily units), no shared keys. Walmart is a second case study |
| Walmart + NOAA | not needed | Walmart already has temperature and other economic columns |

## What to check after `build`

`reports/data_audit_<name>.md` lists every finding with a recommended action. Read the "Issues found" and "What this means for modelling" sections
and put the most important points into the Data section of the final report.
