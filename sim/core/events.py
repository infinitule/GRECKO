"""Structured JSONL event stream — the replay and eval substrate.

EventLog appends events to an in-memory list (and optionally a file).
All consumers (eval, replay, viz) read from this log — no shared state.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, IO, List, Optional

import numpy as np


def _serialise(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


class EventLog:
    def __init__(self, stream: Optional[IO[str]] = None):
        self._events: List[Dict] = []
        self._seq = 0
        self._stream = stream

    def append(self, t: float, event_type: str, **kwargs: Any) -> Dict:
        evt: Dict[str, Any] = {"t": round(t, 6), "seq": self._seq, "type": event_type}
        evt.update(kwargs)
        self._events.append(evt)
        self._seq += 1
        if self._stream is not None:
            self._stream.write(json.dumps(evt, default=_serialise) + "\n")
            self._stream.flush()
        return evt

    @property
    def events(self) -> List[Dict]:
        return self._events

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e, default=_serialise) for e in self._events)

    def write_to(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w") as f:
            f.write(self.to_jsonl())
            if self._events:
                f.write("\n")
