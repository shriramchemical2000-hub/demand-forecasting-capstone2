# Limitations and lessons (read before presenting)

These come from experiments with an earlier version of this project and still apply. Regenerate the numbers with notebook 06 on your own runs.

1. **Compare policies at equal fill rate.** A policy that holds more stock always looks better at one service-level setting. The honest question is how much
   stock each policy needs for the *same* service.
2. **Real shocks are rare, so shock-handling layers show small gains on real data.** On calm real windows they were roughly neutral; they helped most on
   synthetic data where shocks were injected by us. Report real and synthetic results separately.
3. **Calibrated uncertainty gave the clearest and most consistent saving** on real shock windows, but it was modest (a few percent of stock).
4. **Weights tuned on synthetic data transfer badly.** The Critic therefore learns each warning sign's weight from validation errors; a sign that does not predict
   bigger errors is pushed towards zero.
5. **An optional "nowcast" step** (re-centre the plan when recent sales are far from forecast) helped on synthetic surges but not on two real windows, so it is off
   by default (`inventory.nowcast.enabled`).
6. **Intermittent demand and shocks cannot be predicted from the past alone.** Intervals under-cover during a sudden unobserved surge because errors only become
   visible after the forecast horizon.
7. **Costs and lead time are assumptions** (`docs/ASSUMPTIONS.md`). Rankings stayed stable when we varied them; absolute costs did not.
8. **Walmart data is store-level dollars**, so "units" there are dollars of sales volume.
9. **NOAA weather is one station per state** for M5, because M5 hides store locations. Weather has a weak link to the items in the M5 pilot.
10. **The qualitative evaluation needs people.** `scripts/export_trace_review.py` prepares the rating sheet; nobody has rated it until your team does.
