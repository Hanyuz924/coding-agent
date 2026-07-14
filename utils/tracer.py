from __future__ import annotations
 
import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
logger = logging.getLogger("myagent.tracer")

LANGFUSE_MAX_CONTENT = 2000 


def _truncate_for_langfuse(value, limit=LANGFUSE_MAX_CONTENT):
    if isinstance(value, str) and len(value) > limit:
        head, tail = int(limit * 0.7), int(limit * 0.3)
        return f"{value[:head]}\n...[truncated {len(value)-limit} chars]...\n{value[-tail:]}"
    if isinstance(value, dict):
        return {k: _truncate_for_langfuse(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_for_langfuse(v, limit) for v in value]
    return value

class Sink():
    def emit(self, record:dict) -> None:
        raise NotImplementedError
    
class LangfuseSink(Sink):
    def __init__(self) -> None:
        from langfuse import get_client
        self.client =get_client()
    
    def emit(self, record:dict) -> None:
        inputs = record.get("input",{})
        


class JSONLSink(Sink):
    def __init__(self, path:Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sink_file = path
        
    
    def emit(self, record: dict) -> None:
        with open(self.sink_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str))
            f.write("\n")


class Span():
    def __init__(self, span_id, parent_id, span_type, name, inputs):
        self.span_id = span_id
        self.parent_id = parent_id
        self.span_type = span_type        # "turn" | "llm" | "tool"
        self.name = name
        self.inputs = inputs or {}
        self.outputs: dict = {}
        self.status = "ok"
        self.error: Optional[str] = None
        self.start_ts = time.time()
    
    def update_output(self, **kwargs) -> None:
        self.outputs.update(kwargs)

    def to_record(self, trace_id: str) -> dict:
        return {
            "trace_id":    trace_id,
            "span_id":     self.span_id,
            "parent_id":   self.parent_id,
            "type":        self.span_type,
            "name":        self.name,
            "start":       datetime.fromtimestamp(self.start_ts, tz=timezone.utc).isoformat(),
            "duration_ms": round((time.time() - self.start_ts) * 1000),
            "status":      self.status,
            "error":       self.error,
            "inputs":      self.inputs,
            "outputs":     self.outputs,
        }
    
class Tracer:
    def __init__(self, sink:Optional[Sink] = None , trace_dir:Path | None = None):
        self.trace_id = uuid.uuid4().hex[:12]
        if sink is None:
            d = trace_dir or (Path.home() / "coding-agent" / "tmp" / "traces")
            sink = JSONLSink(d / f"{datetime.now():%Y%m%d_%H%M%S}_{self.trace_id}.jsonl")
        self.sink = sink
    @contextmanager
    def span(self, span_type: str, name: str, parent_id:str | None, inputs: Optional[dict] = None) -> Iterator[Span]:
        s = Span(uuid.uuid4().hex[:12], parent_id, span_type, name, inputs)
        try:
            yield s
        except BaseException as e:
            s.status = "error"
            s.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._emit(s)

    def _emit(self, span: Span) -> None:
        self._safe(lambda: self.sink.emit(span.to_record(self.trace_id)))
    
    def event(self, name: str, parent_id: str| None = None, data: Optional[dict] = None) -> None:
        self._safe(lambda: self.sink.emit({
            "trace_id": self.trace_id, "event": name,
            "parent_id": parent_id if parent_id else None,
            "ts": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }))
 
    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as e:
            logger.warning(f"trace emit failed: {e}")