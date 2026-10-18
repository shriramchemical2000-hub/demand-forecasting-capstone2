#!/usr/bin/env python
"""Sample agent reasoning traces into a CSV that teammates can rate (the qualitative evaluation the brief asks for).

    python scripts/export_trace_review.py --results outputs/synthetic --n 6 --out outputs/trace_review.csv
See docs/trace_review.md for the rubric. After rating, run:
    python scripts/export_trace_review.py --summarize outputs/trace_review.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/synthetic")
    ap.add_argument("--n", type=int, default=6, help="events to sample per agent")
    ap.add_argument("--out", default="outputs/trace_review.csv")
    ap.add_argument("--summarize", default=None, help="summarise a rated CSV instead of exporting")
    a = ap.parse_args()

    if a.summarize:
        df = pd.read_csv(a.summarize)
        cols = ["clarity", "correctness", "usefulness"]
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        print(df.groupby("agent")[cols].agg(["mean", "count"]).round(2).to_string())
        print("\nOverall mean:", df[cols].mean().round(2).to_dict())
        return

    tr = pd.read_json(Path(a.results) / "traces.jsonl", lines=True)
    tr = tr[tr["kind"].isin(["thought", "observation", "decision", "revision_request"])]
    sample = pd.concat([g.sample(min(a.n, len(g)), random_state=7) for _, g in tr.groupby("agent")])
    out = sample.sort_values("step")[["step", "agent", "kind", "content"]].copy()
    for c in ("clarity", "correctness", "usefulness", "comment"):
        out[c] = ""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} events to {a.out}. Give each a 1-5 rating for clarity, correctness, usefulness.")


if __name__ == "__main__":
    main()
