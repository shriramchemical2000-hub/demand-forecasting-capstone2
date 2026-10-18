# Qualitative review of agent reasoning traces

The brief asks for qualitative as well as quantitative evaluation. Humans must do this part.

1. `python scripts/export_trace_review.py --results outputs/latest --n 6` creates `outputs/trace_review.csv`.
2. Two or more team members rate every row independently, from 1 (poor) to 5 (excellent):
   - **clarity** - could a planner understand what the agent did and why?
   - **correctness** - is the statement true given the data (check against `forecasts.csv` / `metrics.json`)?
   - **usefulness** - does it help someone decide to trust or challenge the output?
3. Add a short `comment` for anything rated 2 or lower.
4. `python scripts/export_trace_review.py --summarize outputs/trace_review.csv` prints mean ratings per agent.
5. In the report, quote 2-3 traces (best, worst, and the Critic's revision request) and explain what you changed in response.

Ask for the rating to be done on real-data traces, not only synthetic ones.
