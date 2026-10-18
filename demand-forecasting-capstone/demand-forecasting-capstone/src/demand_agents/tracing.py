"""Agent reasoning traces (a hard requirement of the capstone brief).

Every agent writes thoughts, actions, observations and inter-agent messages to
a JSONL file. The dashboard reads this file to show *why* a decision was made.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("thought", "action", "observation", "message", "revision_request", "decision")


def _default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    return str(obj)


class TraceLogger:
    def __init__(self, path=None):
        self.events: list[dict] = []
        self.path = Path(path) if path else None
        self._step = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, agent: str, kind: str, content: str, data: dict | None = None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown trace kind: {kind}")
        self._step += 1
        clean = json.loads(json.dumps(data or {}, default=_default))
        event = {
            "step": self._step,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": agent,
            "kind": kind,
            "content": content,
            "data": clean,
        }
        self.events.append(event)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(event) + "\n")
        return event

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


def read_trace(path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)
