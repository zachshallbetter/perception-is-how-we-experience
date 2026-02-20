"""Structured logging helpers for service-level observability."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict


def utc_timestamp() -> str:
    """Return UTC timestamp in RFC3339 format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def generate_trace_id(prefix: str) -> str:
    """Generate a trace id for a service invocation."""
    return f"{prefix}-{int(time.time() * 1000)}-{os.getpid()}"


def emit_event(
    logger: logging.Logger,
    level: int,
    event: str,
    trace_id: str,
    **fields: Any,
) -> None:
    """Emit a structured JSON log event with stable key ordering."""
    payload: Dict[str, Any] = {
        "event": event,
        "trace_id": trace_id,
        "ts": utc_timestamp(),
    }
    payload.update(fields)
    logger.log(level, json.dumps(payload, sort_keys=True))
