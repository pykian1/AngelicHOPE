from __future__ import annotations
import json
from dataclasses import asdict, fields
from pathlib import Path
from .state import CompressionState, Action, _EXEC

def save_trace(cs: CompressionState, path: str | Path, meta: dict | None = None) -> None: #a run is (checkpoint, meta, ordered actions) packs are recoverable by replay
    blob= {"meta": meta or {}, "history": [asdict(a) for a in cs.history]}
    Path(path).write_text(json.dumps(blob, indent=1))

def load_trace(path: str | Path) -> tuple[dict, list[Action]]:
    blob = json.loads(Path(path).read_text())
    keys = {f.name for f in fields(Action)}
    return blob["meta"], [Action(**{k: v for k, v in d.items() if k in keys}) for d in blob["history"]]

def replay(cs: CompressionState, acts: list[Action], upto: int | None = None) -> CompressionState:
    #cs must be freshly built from the same checkpoint the trace was recorded against
    for a in acts[:upto]:
        _EXEC[a.kind](cs, a)
    return cs